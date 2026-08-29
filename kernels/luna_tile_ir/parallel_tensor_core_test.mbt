///|
fn tc_id(value : Int) -> @luna_tile_ir.LunaTileTensorId raise {
  @luna_tile_ir.LunaTileTensorId::from_stable_int(value)
}

///|
fn tc_tensor(
  id : Int,
  dtype : @luna_tile_ir.LunaTileDType,
  space : @luna_tile_ir.LunaTileAddressSpace,
  shape : ArrayView[Int64],
  strides : ArrayView[Int64],
  offset : Int64,
) -> @luna_tile_ir.LunaTileTensorView raise {
  @luna_tile_ir.LunaTileTensorView::new(
    id=tc_id(id),
    dtype~,
    address_space=space,
    shape~,
    strides_bytes=strides,
    offset_bytes=offset,
    alignment_bytes=16L,
  )
}

///|
fn tc_static_region(
  id : Int,
  extents : ArrayView[Int64],
) -> @luna_tile_ir.LunaTileRegion raise {
  @luna_tile_ir.LunaTileRegion::new(
    tensor=tc_id(id),
    origins=Array::makei(extents.length(), _ => {
      @luna_tile_ir.LunaTileAffineExpr::new(constant=0L, terms=[])
    }),
    extents~,
  )
}

///|
fn tc_global_tile_region(
  id : Int,
  extents : ArrayView[Int64],
) -> @luna_tile_ir.LunaTileRegion raise {
  let origins = Array::makei(extents.length(), index => {
    if index == 0 {
      @luna_tile_ir.LunaTileAffineExpr::new(constant=0L, terms=[
        @luna_tile_ir.LunaTileAffineTerm::new(
          loop_dimension=0,
          coefficient=extents[0],
        ),
      ])
    } else {
      @luna_tile_ir.LunaTileAffineExpr::new(constant=0L, terms=[])
    }
  })
  @luna_tile_ir.LunaTileRegion::new(tensor=tc_id(id), origins~, extents~)
}

///|
fn tensor_core_eligible_program() -> @luna_tile_ir.LunaTileProgram raise {
  let stage = @luna_tile_ir.LunaTileStageId::from_stable_int(1)
  @luna_tile_ir.admit_luna_tile_program(
    @luna_tile_ir.LunaTileProgramInput::new(
      loop_extents=[64L],
      tensors=[
        tc_tensor(1, BFloat16, Global, [1024L, 16L], [32L, 2L], 0L),
        tc_tensor(2, BFloat16, Shared, [16L, 16L], [32L, 2L], 0L),
        tc_tensor(3, BFloat16, Global, [1024L, 16L], [32L, 2L], 0L),
        tc_tensor(4, BFloat16, Shared, [16L, 16L], [32L, 2L], 512L),
        tc_tensor(5, Float32, Global, [16L, 16L], [64L, 4L], 0L),
        tc_tensor(6, Float32, Register, [16L, 16L], [64L, 4L], 0L),
        tc_tensor(7, Float32, Register, [16L, 16L], [64L, 4L], 0L),
        tc_tensor(8, Float32, Global, [1024L, 16L], [64L, 4L], 0L),
      ],
      instructions=[
        LunaTilePipelineStage(stage),
        LunaTileAsyncCopy(
          tc_global_tile_region(1, [16L, 16L]),
          tc_static_region(2, [16L, 16L]),
          stage,
        ),
        LunaTileAsyncCopy(
          tc_global_tile_region(3, [16L, 16L]),
          tc_static_region(4, [16L, 16L]),
          stage,
        ),
        LunaTileBarrier(stage, Block),
        LunaTileCopy(
          tc_static_region(5, [16L, 16L]),
          tc_static_region(6, [16L, 16L]),
        ),
        LunaTileMma(
          destination=tc_id(7),
          left=tc_id(2),
          right=tc_id(4),
          accumulator=tc_id(6),
        ),
        LunaTileCopy(
          tc_static_region(7, [16L, 16L]),
          tc_global_tile_region(8, [16L, 16L]),
        ),
      ],
      constraints=[],
    ),
    @luna_tile_ir.LunaTileLimits::new(
      max_tensors=8,
      max_instructions=7,
      max_rank=3,
      max_loop_rank=1,
      max_constraints=0,
      max_canonical_bytes=32768,
      max_shared_memory_bytes=1024L,
    ),
  )
}

///|
test "tensor-core eligibility remains unqualified and falls back to parallel SIMT" {
  let program = tensor_core_eligible_program()
  let planning = @luna_tile_ir.plan_luna_tile_cuda_aot_input(
    program,
    @luna_tile_ir.LunaTileAotPlanningPolicy::new(
      vector_bytes=4,
      warp_width=32,
      pipeline_stages=1,
      max_shared_memory_bytes=1024L,
    ),
  )
  let limits = @luna_tile_ir.LunaTileCudaLoweringLimits::new(
    max_source_bytes=131072,
  )
  let oracle = @luna_tile_ir.lower_luna_tile_cuda_translation_unit(
    program,
    planning,
    @luna_tile_ir.LunaTileEntryPointId::from_stable_int(80),
    limits,
  )
  let candidate = @luna_tile_ir.specialize_luna_tile_parallel_candidate(
    program,
    planning,
    oracle,
    @luna_tile_ir.LunaTileEntryPointId::from_stable_int(81),
    @luna_tile_ir.LunaTileParallelPolicy::new(
      compute_major=12,
      compute_minor=0,
      block_threads=32,
      block_count=1,
      tile_m=16,
      tile_n=16,
      tile_k=16,
      pipeline_stages=2,
      vector_bytes=4,
      max_shared_memory_bytes=65536L,
      tensor_core_policy=ObserveEligibilityUseParallelSimt,
    ),
    limits,
  )
  assert_eq(candidate.tensor_core_eligibility(), TensorCoreEligibleUnqualified)
  assert_eq(candidate.execution_path(), ParallelSimtCandidate)
  assert_true(
    @utf8.decode(candidate.canonical_bytes()).contains(
      "tensor_core_eligibility=eligible_unqualified",
    ),
  )
  assert_false(@utf8.decode(candidate.source_bytes()).contains("mma.sync"))
}

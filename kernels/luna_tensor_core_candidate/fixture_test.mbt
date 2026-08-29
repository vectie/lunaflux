///|
fn tensor_core_test_id(value : Int) -> @luna_tile_ir.LunaTileTensorId raise {
  @luna_tile_ir.LunaTileTensorId::from_stable_int(value)
}

///|
fn tensor_core_test_tensor(
  id : Int,
  dtype : @luna_tile_ir.LunaTileDType,
  space : @luna_tile_ir.LunaTileAddressSpace,
  shape : ArrayView[Int64],
  strides : ArrayView[Int64],
  offset? : Int64 = 0L,
  alignment? : Int64 = 32L,
) -> @luna_tile_ir.LunaTileTensorView raise {
  @luna_tile_ir.LunaTileTensorView::new(
    id=tensor_core_test_id(id),
    dtype~,
    address_space=space,
    shape~,
    strides_bytes=strides,
    offset_bytes=offset,
    alignment_bytes=alignment,
  )
}

///|
fn tensor_core_test_static_region(
  id : Int,
) -> @luna_tile_ir.LunaTileRegion raise {
  @luna_tile_ir.LunaTileRegion::new(
    tensor=tensor_core_test_id(id),
    origins=[
      @luna_tile_ir.LunaTileAffineExpr::new(constant=0L, terms=[]),
      @luna_tile_ir.LunaTileAffineExpr::new(constant=0L, terms=[]),
    ],
    extents=[16L, 16L],
  )
}

///|
fn tensor_core_test_tiled_region(
  id : Int,
) -> @luna_tile_ir.LunaTileRegion raise {
  @luna_tile_ir.LunaTileRegion::new(
    tensor=tensor_core_test_id(id),
    origins=[
      @luna_tile_ir.LunaTileAffineExpr::new(constant=0L, terms=[
        @luna_tile_ir.LunaTileAffineTerm::new(loop_dimension=0, coefficient=16L),
      ]),
      @luna_tile_ir.LunaTileAffineExpr::new(constant=0L, terms=[]),
    ],
    extents=[16L, 16L],
  )
}

///|
fn tensor_core_test_program(
  tile_count? : Int64 = 64L,
  barrier_scope? : @luna_tile_ir.LunaTileBarrierScope = Block,
  global_alignment? : Int64 = 32L,
) -> @luna_tile_ir.LunaTileProgram raise {
  let stage = @luna_tile_ir.LunaTileStageId::from_stable_int(1)
  let rows = tile_count * 16L
  @luna_tile_ir.admit_luna_tile_program(
    @luna_tile_ir.LunaTileProgramInput::new(
      loop_extents=[tile_count],
      tensors=[
        tensor_core_test_tensor(
          1,
          BFloat16,
          Global,
          [rows, 16L],
          [32L, 2L],
          alignment=global_alignment,
        ),
        tensor_core_test_tensor(
          2,
          BFloat16,
          Shared,
          [16L, 16L],
          [32L, 2L],
          alignment=16L,
        ),
        tensor_core_test_tensor(
          3,
          BFloat16,
          Global,
          [rows, 16L],
          [32L, 2L],
          alignment=global_alignment,
        ),
        tensor_core_test_tensor(
          4,
          BFloat16,
          Shared,
          [16L, 16L],
          [32L, 2L],
          offset=512L,
          alignment=16L,
        ),
        tensor_core_test_tensor(
          5,
          Float32,
          Global,
          [16L, 16L],
          [64L, 4L],
          alignment=global_alignment,
        ),
        tensor_core_test_tensor(6, Float32, Register, [16L, 16L], [64L, 4L]),
        tensor_core_test_tensor(7, Float32, Register, [16L, 16L], [64L, 4L]),
        tensor_core_test_tensor(
          8,
          Float32,
          Global,
          [rows, 16L],
          [64L, 4L],
          alignment=global_alignment,
        ),
      ],
      instructions=[
        LunaTilePipelineStage(stage),
        LunaTileAsyncCopy(
          tensor_core_test_tiled_region(1),
          tensor_core_test_static_region(2),
          stage,
        ),
        LunaTileAsyncCopy(
          tensor_core_test_tiled_region(3),
          tensor_core_test_static_region(4),
          stage,
        ),
        LunaTileBarrier(stage, barrier_scope),
        LunaTileCopy(
          tensor_core_test_static_region(5),
          tensor_core_test_static_region(6),
        ),
        LunaTileMma(
          destination=tensor_core_test_id(7),
          left=tensor_core_test_id(2),
          right=tensor_core_test_id(4),
          accumulator=tensor_core_test_id(6),
        ),
        LunaTileCopy(
          tensor_core_test_static_region(7),
          tensor_core_test_tiled_region(8),
        ),
      ],
      constraints=[],
    ),
    @luna_tile_ir.LunaTileLimits::new(
      max_tensors=8,
      max_instructions=7,
      max_rank=2,
      max_loop_rank=1,
      max_constraints=0,
      max_canonical_bytes=32768,
      max_shared_memory_bytes=1024L,
    ),
  )
}

///|
fn tensor_core_test_serial(
  program : @luna_tile_ir.LunaTileProgram,
) -> @luna_tile_ir.LunaTileCudaTranslationUnit raise {
  let planning = @luna_tile_ir.plan_luna_tile_cuda_aot_input(
    program,
    @luna_tile_ir.LunaTileAotPlanningPolicy::new(
      vector_bytes=4,
      warp_width=32,
      pipeline_stages=1,
      max_shared_memory_bytes=1024L,
    ),
  )
  @luna_tile_ir.lower_luna_tile_cuda_translation_unit(
    program,
    planning,
    @luna_tile_ir.LunaTileEntryPointId::from_stable_int(1301),
    @luna_tile_ir.LunaTileCudaLoweringLimits::new(max_source_bytes=131072),
  )
}

///|
fn tensor_core_test_parallel(
  program : @luna_tile_ir.LunaTileProgram,
  serial : @luna_tile_ir.LunaTileCudaTranslationUnit,
  compute_major? : Int = 12,
  compute_minor? : Int = 0,
) -> @luna_tile_ir.LunaTileParallelCandidate raise {
  @luna_tile_ir.specialize_luna_tile_parallel_candidate(
    program,
    @luna_tile_ir.plan_luna_tile_cuda_aot_input(
      program,
      @luna_tile_ir.LunaTileAotPlanningPolicy::new(
        vector_bytes=4,
        warp_width=32,
        pipeline_stages=1,
        max_shared_memory_bytes=1024L,
      ),
    ),
    serial,
    @luna_tile_ir.LunaTileEntryPointId::from_stable_int(1302),
    @luna_tile_ir.LunaTileParallelPolicy::new(
      compute_major~,
      compute_minor~,
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
    @luna_tile_ir.LunaTileCudaLoweringLimits::new(max_source_bytes=131072),
  )
}

///|
fn tensor_core_test_contract(
  compute_major? : Int = 12,
  compute_minor? : Int = 0,
  m? : Int = 16,
  right_layout? : @luna_tensor_core_candidate.LunaTileTensorCoreMatrixLayout = TensorCoreRowMajor,
  accumulator? : @luna_tensor_core_candidate.LunaTileTensorCoreAccumulator = TensorCoreAccumulatorF32,
  alignment? : Int = 32,
  epilogue? : @luna_tensor_core_candidate.LunaTileTensorCoreEpilogue = TensorCoreIdentityEpilogue,
) -> @luna_tensor_core_candidate.LunaTileBf16Mma16x16x16Contract raise {
  @luna_tensor_core_candidate.LunaTileBf16Mma16x16x16Contract::new(
    compute_major~,
    compute_minor~,
    m~,
    n=16,
    k=16,
    left_layout=TensorCoreRowMajor,
    right_layout~,
    accumulator~,
    output_layout=TensorCoreRowMajor,
    left_alignment_bytes=alignment,
    right_alignment_bytes=alignment,
    accumulator_alignment_bytes=alignment,
    output_alignment_bytes=alignment,
    epilogue~,
    block_threads=32,
    maximum_registers_per_thread=128,
  )
}

///|
fn tensor_core_test_limits(
  source_bytes? : Int = 131072,
) -> @luna_tensor_core_candidate.LunaTileTensorCoreCandidateLimits raise {
  @luna_tensor_core_candidate.LunaTileTensorCoreCandidateLimits::new(
    maximum_source_bytes=source_bytes,
    maximum_canonical_bytes=131072,
  )
}

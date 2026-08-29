///|
fn cross_id(value : Int) -> @luna_tile_ir.LunaTileTensorId raise {
  @luna_tile_ir.LunaTileTensorId::from_stable_int(value)
}

///|
fn cross_tensor(
  id : Int,
  space : @luna_tile_ir.LunaTileAddressSpace,
  shape : ArrayView[Int64],
) -> @luna_tile_ir.LunaTileTensorView raise {
  @luna_tile_ir.LunaTileTensorView::new(
    id=cross_id(id),
    dtype=BFloat16,
    address_space=space,
    shape~,
    strides_bytes=[32L, 2L],
    offset_bytes=0L,
    alignment_bytes=16L,
  )
}

///|
fn cross_region(id : Int, swapped : Bool) -> @luna_tile_ir.LunaTileRegion raise {
  let outer = if swapped { 1 } else { 0 }
  let inner = if swapped { 0 } else { 1 }
  @luna_tile_ir.LunaTileRegion::new(
    tensor=cross_id(id),
    origins=[
      @luna_tile_ir.LunaTileAffineExpr::new(constant=0L, terms=[
        @luna_tile_ir.LunaTileAffineTerm::new(
          loop_dimension=outer,
          coefficient=1L,
        ),
      ]),
      @luna_tile_ir.LunaTileAffineExpr::new(constant=0L, terms=[
        @luna_tile_ir.LunaTileAffineTerm::new(
          loop_dimension=inner,
          coefficient=2L,
        ),
      ]),
    ],
    extents=[1L, 2L],
  )
}

///|
fn cross_static_region(id : Int) -> @luna_tile_ir.LunaTileRegion raise {
  @luna_tile_ir.LunaTileRegion::new(
    tensor=cross_id(id),
    origins=[
      @luna_tile_ir.LunaTileAffineExpr::new(constant=0L, terms=[]),
      @luna_tile_ir.LunaTileAffineExpr::new(constant=0L, terms=[]),
    ],
    extents=[1L, 2L],
  )
}

///|
test "parallel specialization rejects cross-instruction global write overlap" {
  let stage = @luna_tile_ir.LunaTileStageId::from_stable_int(1)
  let program = @luna_tile_ir.admit_luna_tile_program(
    @luna_tile_ir.LunaTileProgramInput::new(
      loop_extents=[8L, 8L],
      tensors=[
        cross_tensor(1, Global, [8L, 16L]),
        cross_tensor(2, Shared, [1L, 2L]),
        cross_tensor(3, Register, [1L, 2L]),
        cross_tensor(4, Global, [8L, 16L]),
      ],
      instructions=[
        LunaTilePipelineStage(stage),
        LunaTileAsyncCopy(cross_region(1, false), cross_static_region(2), stage),
        LunaTileBarrier(stage, Block),
        LunaTileCopy(cross_static_region(2), cross_static_region(3)),
        LunaTileCopy(cross_static_region(3), cross_region(4, false)),
        LunaTileCopy(cross_static_region(3), cross_region(4, true)),
      ],
      constraints=[],
    ),
    @luna_tile_ir.LunaTileLimits::new(
      max_tensors=4,
      max_instructions=6,
      max_rank=2,
      max_loop_rank=2,
      max_constraints=0,
      max_canonical_bytes=16384,
      max_shared_memory_bytes=4L,
    ),
  )
  let planning = @luna_tile_ir.plan_luna_tile_cuda_aot_input(
    program,
    @luna_tile_ir.LunaTileAotPlanningPolicy::new(
      vector_bytes=4,
      warp_width=32,
      pipeline_stages=1,
      max_shared_memory_bytes=4L,
    ),
  )
  let limits = @luna_tile_ir.LunaTileCudaLoweringLimits::new(
    max_source_bytes=65536,
  )
  let oracle = @luna_tile_ir.lower_luna_tile_cuda_translation_unit(
    program,
    planning,
    @luna_tile_ir.LunaTileEntryPointId::from_stable_int(90),
    limits,
  )
  try
    @luna_tile_ir.specialize_luna_tile_parallel_candidate(
      program,
      planning,
      oracle,
      @luna_tile_ir.LunaTileEntryPointId::from_stable_int(91),
      @luna_tile_ir.LunaTileParallelPolicy::new(
        compute_major=12,
        compute_minor=0,
        block_threads=32,
        block_count=1,
        tile_m=1,
        tile_n=2,
        tile_k=1,
        pipeline_stages=2,
        vector_bytes=4,
        max_shared_memory_bytes=4096L,
        tensor_core_policy=ObserveEligibilityUseParallelSimt,
      ),
      limits,
    )
  catch {
    InvalidProgram(ParallelMapping) => ()
    _ => fail("unexpected cross-instruction overlap failure")
  } noraise {
    _ => fail("cross-instruction global write overlap was accepted")
  }
}

///|
test "parallel specialization rejects cross-worker global read-write alias" {
  let stage = @luna_tile_ir.LunaTileStageId::from_stable_int(1)
  let program = @luna_tile_ir.admit_luna_tile_program(
    @luna_tile_ir.LunaTileProgramInput::new(
      loop_extents=[8L, 8L],
      tensors=[
        cross_tensor(1, Global, [8L, 16L]),
        cross_tensor(2, Shared, [1L, 2L]),
        cross_tensor(3, Register, [1L, 2L]),
        cross_tensor(4, Global, [8L, 16L]),
      ],
      instructions=[
        LunaTilePipelineStage(stage),
        LunaTileAsyncCopy(cross_region(1, true), cross_static_region(2), stage),
        LunaTileBarrier(stage, Block),
        LunaTileCopy(cross_static_region(2), cross_static_region(3)),
        LunaTileCopy(cross_static_region(3), cross_region(1, false)),
        LunaTileCopy(cross_static_region(3), cross_region(4, false)),
      ],
      constraints=[],
    ),
    @luna_tile_ir.LunaTileLimits::new(
      max_tensors=4,
      max_instructions=6,
      max_rank=2,
      max_loop_rank=2,
      max_constraints=0,
      max_canonical_bytes=16384,
      max_shared_memory_bytes=4L,
    ),
  )
  let planning = @luna_tile_ir.plan_luna_tile_cuda_aot_input(
    program,
    @luna_tile_ir.LunaTileAotPlanningPolicy::new(
      vector_bytes=4,
      warp_width=32,
      pipeline_stages=1,
      max_shared_memory_bytes=4L,
    ),
  )
  let limits = @luna_tile_ir.LunaTileCudaLoweringLimits::new(
    max_source_bytes=65536,
  )
  let oracle = @luna_tile_ir.lower_luna_tile_cuda_translation_unit(
    program,
    planning,
    @luna_tile_ir.LunaTileEntryPointId::from_stable_int(92),
    limits,
  )
  try
    @luna_tile_ir.specialize_luna_tile_parallel_candidate(
      program,
      planning,
      oracle,
      @luna_tile_ir.LunaTileEntryPointId::from_stable_int(93),
      @luna_tile_ir.LunaTileParallelPolicy::new(
        compute_major=12,
        compute_minor=0,
        block_threads=32,
        block_count=2,
        tile_m=1,
        tile_n=2,
        tile_k=1,
        pipeline_stages=2,
        vector_bytes=4,
        max_shared_memory_bytes=4096L,
        tensor_core_policy=ObserveEligibilityUseParallelSimt,
      ),
      limits,
    )
  catch {
    InvalidProgram(ParallelMapping) => ()
    _ => fail("unexpected global read-write alias failure")
  } noraise {
    _ => fail("cross-worker global read-write alias was accepted")
  }
}

///|
fn parallel_id(id : Int) -> @luna_tile_ir.LunaTileTensorId raise {
  @luna_tile_ir.LunaTileTensorId::from_stable_int(id)
}

///|
fn parallel_tensor(
  id : Int,
  space : @luna_tile_ir.LunaTileAddressSpace,
  shape : ArrayView[Int64],
  strides : ArrayView[Int64],
  offset : Int64,
) -> @luna_tile_ir.LunaTileTensorView raise {
  @luna_tile_ir.LunaTileTensorView::new(
    id=parallel_id(id),
    dtype=BFloat16,
    address_space=space,
    shape~,
    strides_bytes=strides,
    offset_bytes=offset,
    alignment_bytes=16L,
  )
}

///|
fn parallel_region(
  id : Int,
  global_tile : Bool,
) -> @luna_tile_ir.LunaTileRegion raise {
  let first = if global_tile {
    @luna_tile_ir.LunaTileAffineExpr::new(constant=0L, terms=[
      @luna_tile_ir.LunaTileAffineTerm::new(loop_dimension=0, coefficient=1L),
    ])
  } else {
    @luna_tile_ir.LunaTileAffineExpr::new(constant=0L, terms=[])
  }
  @luna_tile_ir.LunaTileRegion::new(
    tensor=parallel_id(id),
    origins=[
      first,
      @luna_tile_ir.LunaTileAffineExpr::new(constant=0L, terms=[]),
    ],
    extents=[1L, 2L],
  )
}

///|
fn parallel_program(
  output_disjoint : Bool,
  loop_extent? : Int64 = 64L,
) -> @luna_tile_ir.LunaTileProgram raise {
  let stage = @luna_tile_ir.LunaTileStageId::from_stable_int(1)
  @luna_tile_ir.admit_luna_tile_program(
    @luna_tile_ir.LunaTileProgramInput::new(
      loop_extents=[loop_extent],
      tensors=[
        parallel_tensor(1, Global, [loop_extent, 2L], [4L, 2L], 0L),
        parallel_tensor(2, Shared, [1L, 2L], [4L, 2L], 0L),
        parallel_tensor(3, Register, [1L, 2L], [4L, 2L], 0L),
        parallel_tensor(4, Global, [loop_extent, 2L], [4L, 2L], 0L),
        parallel_tensor(5, Global, [loop_extent, 2L], [4L, 2L], 0L),
        parallel_tensor(6, Shared, [1L, 2L], [4L, 2L], 16L),
        parallel_tensor(7, Register, [1L, 2L], [4L, 2L], 0L),
        parallel_tensor(8, Global, [loop_extent, 2L], [4L, 2L], 0L),
      ],
      instructions=[
        LunaTilePipelineStage(stage),
        LunaTileAsyncCopy(
          parallel_region(1, true),
          parallel_region(2, false),
          stage,
        ),
        LunaTileBarrier(stage, Block),
        LunaTileCopy(parallel_region(2, false), parallel_region(3, false)),
        LunaTileCopy(
          parallel_region(3, false),
          parallel_region(4, output_disjoint),
        ),
        LunaTileCopy(parallel_region(5, true), parallel_region(6, false)),
        LunaTileCopy(parallel_region(6, false), parallel_region(7, false)),
        @luna_tile_ir.LunaTileInstruction::elementwise(Silu, parallel_id(7), [
          parallel_id(7),
        ]),
        LunaTileCopy(
          parallel_region(7, false),
          parallel_region(8, output_disjoint),
        ),
      ],
      constraints=[],
    ),
    @luna_tile_ir.LunaTileLimits::new(
      max_tensors=8,
      max_instructions=9,
      max_rank=2,
      max_loop_rank=1,
      max_constraints=0,
      max_canonical_bytes=32768,
      max_shared_memory_bytes=64L,
    ),
  )
}

///|
fn parallel_planning(
  program : @luna_tile_ir.LunaTileProgram,
) -> @luna_tile_ir.LunaTileCudaAotInput raise {
  @luna_tile_ir.plan_luna_tile_cuda_aot_input(
    program,
    @luna_tile_ir.LunaTileAotPlanningPolicy::new(
      vector_bytes=4,
      warp_width=32,
      pipeline_stages=1,
      max_shared_memory_bytes=64L,
    ),
  )
}

///|
fn parallel_oracle(
  program : @luna_tile_ir.LunaTileProgram,
  planning : @luna_tile_ir.LunaTileCudaAotInput,
) -> @luna_tile_ir.LunaTileCudaTranslationUnit raise {
  @luna_tile_ir.lower_luna_tile_cuda_translation_unit(
    program,
    planning,
    @luna_tile_ir.LunaTileEntryPointId::from_stable_int(70),
    @luna_tile_ir.LunaTileCudaLoweringLimits::new(max_source_bytes=65536),
  )
}

///|
fn parallel_policy(
  tensor_core_policy? : @luna_tile_ir.LunaTileTensorCorePolicy = ObserveEligibilityUseParallelSimt,
) -> @luna_tile_ir.LunaTileParallelPolicy raise {
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
    max_shared_memory_bytes=49152L,
    tensor_core_policy~,
  )
}

///|
test "parallel specialization binds mapping reuse overlap and oracle deterministically" {
  let program = parallel_program(true)
  let planning = parallel_planning(program)
  let oracle = parallel_oracle(program, planning)
  let entry = @luna_tile_ir.LunaTileEntryPointId::from_stable_int(71)
  let limits = @luna_tile_ir.LunaTileCudaLoweringLimits::new(
    max_source_bytes=65536,
  )
  let first = @luna_tile_ir.specialize_luna_tile_parallel_candidate(
    program,
    planning,
    oracle,
    entry,
    parallel_policy(),
    limits,
  )
  let repeated = @luna_tile_ir.specialize_luna_tile_parallel_candidate(
    program,
    planning,
    oracle,
    entry,
    parallel_policy(),
    limits,
  )
  assert_eq(first.digest(), repeated.digest())
  assert_eq(first.source_bytes(), repeated.source_bytes())
  assert_eq(first.canonical_bytes(), repeated.canonical_bytes())
  assert_eq(first.program_digest(), program.digest())
  assert_eq(first.planning_digest(), planning.digest())
  assert_eq(first.serial_oracle_digest(), oracle.digest())
  assert_eq(first.execution_path(), ParallelSimtCandidate)
  assert_eq(first.tensor_core_eligibility(), TensorCoreNoMma)
  assert_eq(first.mapping().block_threads(), 32)
  assert_eq(first.mapping().warps_per_block(), 1)
  assert_eq(first.mapping().rounds_per_worker(), 2L)
  assert_eq(first.shared_plan().arena_bytes(), 16L)
  assert_eq(first.shared_plan().total_shared_memory_bytes(), 1024L)
  assert_eq(first.shared_plan().reuse_count(), 1)
  let source = @utf8.decode(first.source_bytes())
  for
    required in [
      "blockDim.x != 32", "luna_pipe < 3LL", "__pipeline_commit()", "__pipeline_wait_prior(0)",
      "luna_compute_round % 2LL", "luna_warp * 32LL", "luna_lane", "__syncthreads()",
    ] {
    assert_true(source.contains(required))
  }
  assert_false(source.contains("lunaflux_lunatile_v1_ep_70"))
  let canonical = @utf8.decode(first.canonical_bytes())
  for
    required in [
      "serial_oracle_role=qualification_only", "target=sm_120", "execution_path=parallel_simt_candidate",
      "tensor_core_eligibility=no_mma", "device_authority=none", "manifest_bindable=false",
    ] {
    assert_true(canonical.contains(required))
  }
}

///|
test "parallel specialization rejects overlap substitution and asserted qualification" {
  let limits = @luna_tile_ir.LunaTileCudaLoweringLimits::new(
    max_source_bytes=65536,
  )
  let entry = @luna_tile_ir.LunaTileEntryPointId::from_stable_int(71)
  let overlap = parallel_program(false)
  let overlap_plan = parallel_planning(overlap)
  try
    @luna_tile_ir.specialize_luna_tile_parallel_candidate(
      overlap,
      overlap_plan,
      parallel_oracle(overlap, overlap_plan),
      entry,
      parallel_policy(),
      limits,
    )
  catch {
    @luna_tile_ir.InvalidProgram(ParallelMapping) => ()
    _ => fail("unexpected overlapping-output failure")
  } noraise {
    _ => fail("overlapping parallel output was accepted")
  }
  let uneven = parallel_program(true, loop_extent=63L)
  let uneven_plan = parallel_planning(uneven)
  try
    @luna_tile_ir.specialize_luna_tile_parallel_candidate(
      uneven,
      uneven_plan,
      parallel_oracle(uneven, uneven_plan),
      entry,
      parallel_policy(),
      limits,
    )
  catch {
    @luna_tile_ir.InvalidProgram(ParallelMapping) => ()
    _ => fail("unexpected uneven-mapping failure")
  } noraise {
    _ => fail("non-divisible tile mapping was accepted")
  }
  let program = parallel_program(true)
  let planning = parallel_planning(program)
  let foreign = parallel_program(true, loop_extent=96L)
  let foreign_plan = parallel_planning(foreign)
  try
    @luna_tile_ir.specialize_luna_tile_parallel_candidate(
      program,
      planning,
      parallel_oracle(foreign, foreign_plan),
      entry,
      parallel_policy(),
      limits,
    )
  catch {
    @luna_tile_ir.InvalidProgram(ParallelPolicy) => ()
    _ => fail("unexpected serial-oracle substitution failure")
  } noraise {
    _ => fail("foreign serial oracle was accepted")
  }
  try
    @luna_tile_ir.specialize_luna_tile_parallel_candidate(
      program,
      planning,
      parallel_oracle(program, planning),
      entry,
      parallel_policy(tensor_core_policy=RequireExternallyQualifiedTensorCore),
      limits,
    )
  catch {
    @luna_tile_ir.InvalidProgram(TensorCoreQualification) => ()
    _ => fail("unexpected tensor-core qualification failure")
  } noraise {
    _ => fail("caller-selected tensor-core qualification was accepted")
  }
  try
    @luna_tile_ir.specialize_luna_tile_parallel_candidate(
      program,
      planning,
      parallel_oracle(program, planning),
      entry,
      parallel_policy(),
      @luna_tile_ir.LunaTileCudaLoweringLimits::new(max_source_bytes=32),
    )
  catch {
    @luna_tile_ir.InvalidProgram(CudaLowering) => ()
    _ => fail("unexpected source-bound failure")
  } noraise {
    _ => fail("unbounded parallel source was accepted")
  }
}

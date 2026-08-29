///|
fn generic_cuda_tensor(
  id : Int,
  dtype : @luna_tile_ir.LunaTileDType,
  space : @luna_tile_ir.LunaTileAddressSpace,
  shape : ArrayView[Int64],
  strides : ArrayView[Int64],
  offset : Int64,
  alignment : Int64,
) -> @luna_tile_ir.LunaTileTensorView raise {
  @luna_tile_ir.LunaTileTensorView::new(
    id=@luna_tile_ir.LunaTileTensorId::from_stable_int(id),
    dtype~,
    address_space=space,
    shape~,
    strides_bytes=strides,
    offset_bytes=offset,
    alignment_bytes=alignment,
  )
}

///|
fn generic_cuda_region(
  id : Int,
  shape : ArrayView[Int64],
) -> @luna_tile_ir.LunaTileRegion raise {
  let origins = Array::makei(shape.length(), _ => {
    @luna_tile_ir.LunaTileAffineExpr::new(constant=0L, terms=[])
  })
  @luna_tile_ir.LunaTileRegion::new(
    tensor=@luna_tile_ir.LunaTileTensorId::from_stable_int(id),
    origins~,
    extents=shape,
  )
}

///|
fn generic_cuda_id(id : Int) -> @luna_tile_ir.LunaTileTensorId raise {
  @luna_tile_ir.LunaTileTensorId::from_stable_int(id)
}

///|
fn generic_cuda_program(
  last_kind? : @luna_tile_ir.LunaTileElementwiseKind = Silu,
) -> @luna_tile_ir.LunaTileProgram raise {
  let stage = @luna_tile_ir.LunaTileStageId::from_stable_int(1)
  let instructions : Array[@luna_tile_ir.LunaTileInstruction] = [
    LunaTilePipelineStage(stage),
    LunaTileAsyncCopy(
      generic_cuda_region(1, [2L, 2L]),
      generic_cuda_region(2, [2L, 2L]),
      stage,
    ),
    LunaTileAsyncCopy(
      generic_cuda_region(4, [2L, 2L]),
      generic_cuda_region(5, [2L, 2L]),
      stage,
    ),
    LunaTileBarrier(stage, Block),
    LunaTileCopy(
      generic_cuda_region(2, [2L, 2L]),
      generic_cuda_region(3, [2L, 2L]),
    ),
    LunaTileCopy(
      generic_cuda_region(5, [2L, 2L]),
      generic_cuda_region(6, [2L, 2L]),
    ),
    LunaTileCopy(
      generic_cuda_region(7, [2L, 2L]),
      generic_cuda_region(8, [2L, 2L]),
    ),
    LunaTileMma(
      destination=generic_cuda_id(9),
      left=generic_cuda_id(3),
      right=generic_cuda_id(6),
      accumulator=generic_cuda_id(8),
    ),
    LunaTileReduce(
      Sum,
      destination=generic_cuda_id(10),
      source=generic_cuda_id(9),
      axis=1,
    ),
    LunaTileReduce(
      Maximum,
      destination=generic_cuda_id(10),
      source=generic_cuda_id(9),
      axis=1,
    ),
    @luna_tile_ir.LunaTileInstruction::elementwise(Add, generic_cuda_id(11), [
      generic_cuda_id(10),
      generic_cuda_id(10),
    ]),
    @luna_tile_ir.LunaTileInstruction::elementwise(
      Multiply,
      generic_cuda_id(11),
      [generic_cuda_id(11), generic_cuda_id(10)],
    ),
    @luna_tile_ir.LunaTileInstruction::elementwise(
      Maximum,
      generic_cuda_id(11),
      [generic_cuda_id(11), generic_cuda_id(10)],
    ),
  ]
  instructions.push(
    match last_kind {
      Silu =>
        @luna_tile_ir.LunaTileInstruction::elementwise(
          Silu,
          generic_cuda_id(11),
          [generic_cuda_id(11)],
        )
      Add | Multiply | Maximum =>
        @luna_tile_ir.LunaTileInstruction::elementwise(
          last_kind,
          generic_cuda_id(11),
          [generic_cuda_id(11), generic_cuda_id(10)],
        )
    },
  )
  instructions.push(
    LunaTileCopy(
      generic_cuda_region(11, [2L, 1L]),
      generic_cuda_region(12, [2L, 1L]),
    ),
  )
  @luna_tile_ir.admit_luna_tile_program(
    @luna_tile_ir.LunaTileProgramInput::new(
      loop_extents=[1L],
      tensors=[
        generic_cuda_tensor(1, BFloat16, Global, [2L, 2L], [4L, 2L], 0L, 16L),
        generic_cuda_tensor(2, BFloat16, Shared, [2L, 2L], [4L, 2L], 0L, 16L),
        generic_cuda_tensor(3, BFloat16, Register, [2L, 2L], [4L, 2L], 0L, 16L),
        generic_cuda_tensor(4, BFloat16, Global, [2L, 2L], [4L, 2L], 0L, 16L),
        generic_cuda_tensor(5, BFloat16, Shared, [2L, 2L], [4L, 2L], 16L, 16L),
        generic_cuda_tensor(6, BFloat16, Register, [2L, 2L], [4L, 2L], 0L, 16L),
        generic_cuda_tensor(7, Float32, Global, [2L, 2L], [8L, 4L], 0L, 16L),
        generic_cuda_tensor(8, Float32, Register, [2L, 2L], [8L, 4L], 0L, 16L),
        generic_cuda_tensor(9, Float32, Register, [2L, 2L], [8L, 4L], 0L, 16L),
        generic_cuda_tensor(10, Float32, Register, [2L, 1L], [4L, 4L], 0L, 16L),
        generic_cuda_tensor(11, Float32, Register, [2L, 1L], [4L, 4L], 0L, 16L),
        generic_cuda_tensor(12, Float32, Global, [2L, 1L], [4L, 4L], 0L, 16L),
      ],
      instructions~,
      constraints=[],
    ),
    @luna_tile_ir.LunaTileLimits::new(
      max_tensors=12,
      max_instructions=16,
      max_rank=2,
      max_loop_rank=1,
      max_constraints=0,
      max_canonical_bytes=32768,
      max_shared_memory_bytes=64L,
    ),
  )
}

///|
fn generic_cuda_planning(
  program : @luna_tile_ir.LunaTileProgram,
) -> @luna_tile_ir.LunaTileCudaAotInput raise {
  @luna_tile_ir.plan_luna_tile_cuda_aot_input(
    program,
    @luna_tile_ir.LunaTileAotPlanningPolicy::new(
      vector_bytes=16,
      warp_width=32,
      pipeline_stages=2,
      max_shared_memory_bytes=64L,
    ),
  )
}

///|
test "generic typed CUDA lowering covers the complete LunaTile operation set" {
  let program = generic_cuda_program()
  let planning = generic_cuda_planning(program)
  let operations = program.operation_summary()
  assert_eq(operations.copy_count(), 4)
  assert_eq(operations.async_copy_count(), 2)
  assert_eq(operations.barrier_count(), 1)
  assert_eq(operations.pipeline_stage_count(), 1)
  assert_eq(operations.mma_count(), 1)
  assert_eq(operations.reduction_count(), 2)
  assert_eq(operations.elementwise_count(), 4)
  let entry = @luna_tile_ir.LunaTileEntryPointId::from_stable_int(41)
  let limits = @luna_tile_ir.LunaTileCudaLoweringLimits::new(
    max_source_bytes=65536,
  )
  let first = @luna_tile_ir.lower_luna_tile_cuda_translation_unit(
    program, planning, entry, limits,
  )
  let repeated = @luna_tile_ir.lower_luna_tile_cuda_translation_unit(
    generic_cuda_program(),
    generic_cuda_planning(generic_cuda_program()),
    entry,
    limits,
  )
  assert_eq(first.digest(), repeated.digest())
  assert_eq(first.source_bytes(), repeated.source_bytes())
  assert_eq(first.source_digest(), repeated.source_digest())
  assert_eq(first.canonical_bytes(), repeated.canonical_bytes())
  assert_eq(first.program_digest(), program.digest())
  assert_eq(first.planning_digest(), planning.digest())
  assert_eq(first.entry_point(), "lunaflux_lunatile_v1_ep_41")
  assert_eq(first.memory_plan().vector_bytes(), 16)
  assert_eq(first.memory_plan().pipeline_stages(), 1)
  let source = @utf8.decode(first.source_bytes())
  for
    required in [
      "__pipeline_memcpy_async", "__pipeline_wait_prior(0)", "__syncthreads()", "fmaf(",
      "fmaxf(", "expf(", "reinterpret_cast<uint32_t*>",
    ] {
    assert_true(source.contains(required))
  }
  assert_false(source.contains("residual"))
  assert_false(source.contains("llama"))
  let canonical = @utf8.decode(first.canonical_bytes())
  assert_true(canonical.contains("compiler_authority=none"))
  assert_true(canonical.contains("runtime_authority=none"))
}

///|
test "generic CUDA lowering rejects plan substitution and unbounded output" {
  let program = generic_cuda_program()
  let foreign = generic_cuda_program(last_kind=Add)
  let entry = @luna_tile_ir.LunaTileEntryPointId::from_stable_int(1)
  try
    @luna_tile_ir.lower_luna_tile_cuda_translation_unit(
      program,
      generic_cuda_planning(foreign),
      entry,
      @luna_tile_ir.LunaTileCudaLoweringLimits::new(max_source_bytes=65536),
    )
  catch {
    @luna_tile_ir.InvalidProgram(CudaLowering) => ()
    _ => fail("unexpected substituted planning failure")
  } noraise {
    _ => fail("foreign planning input was accepted")
  }
  try
    @luna_tile_ir.lower_luna_tile_cuda_translation_unit(
      program,
      generic_cuda_planning(program),
      entry,
      @luna_tile_ir.LunaTileCudaLoweringLimits::new(max_source_bytes=32),
    )
  catch {
    @luna_tile_ir.InvalidProgram(CudaLowering) => ()
    _ => fail("unexpected source-bound failure")
  } noraise {
    _ => fail("oversized CUDA source was accepted")
  }
}

///|
test "elementwise arity and async vector planning fail closed" {
  let output = generic_cuda_id(1)
  try
    @luna_tile_ir.admit_luna_tile_program(
      @luna_tile_ir.LunaTileProgramInput::new(
        loop_extents=[1L],
        tensors=[generic_cuda_tensor(1, Float32, Register, [1L], [4L], 0L, 4L)],
        instructions=[
          @luna_tile_ir.LunaTileInstruction::elementwise(Add, output, [output]),
        ],
        constraints=[],
      ),
      @luna_tile_ir.LunaTileLimits::new(
        max_tensors=1,
        max_instructions=1,
        max_rank=1,
        max_loop_rank=1,
        max_constraints=0,
        max_canonical_bytes=4096,
        max_shared_memory_bytes=0L,
      ),
    )
  catch {
    @luna_tile_ir.InvalidProgram(Elementwise) => ()
    _ => fail("unexpected elementwise arity failure")
  } noraise {
    _ => fail("invalid elementwise arity was admitted")
  }
  let stage = @luna_tile_ir.LunaTileStageId::from_stable_int(1)
  let narrow = @luna_tile_ir.admit_luna_tile_program(
    @luna_tile_ir.LunaTileProgramInput::new(
      loop_extents=[1L],
      tensors=[
        generic_cuda_tensor(1, BFloat16, Global, [2L], [2L], 0L, 2L),
        generic_cuda_tensor(2, BFloat16, Shared, [2L], [2L], 0L, 2L),
      ],
      instructions=[
        LunaTilePipelineStage(stage),
        LunaTileAsyncCopy(
          generic_cuda_region(1, [2L]),
          generic_cuda_region(2, [2L]),
          stage,
        ),
        LunaTileBarrier(stage, Block),
      ],
      constraints=[],
    ),
    @luna_tile_ir.LunaTileLimits::new(
      max_tensors=2,
      max_instructions=3,
      max_rank=1,
      max_loop_rank=1,
      max_constraints=0,
      max_canonical_bytes=4096,
      max_shared_memory_bytes=4L,
    ),
  )
  try
    @luna_tile_ir.plan_luna_tile_cuda_aot_input(
      narrow,
      @luna_tile_ir.LunaTileAotPlanningPolicy::new(
        vector_bytes=4,
        warp_width=32,
        pipeline_stages=1,
        max_shared_memory_bytes=4L,
      ),
    )
  catch {
    InvalidProgram(AotPlanningPolicy) => ()
    _ => fail("unexpected async-vector planning failure")
  } noraise {
    _ => fail("unaligned asynchronous copy was planned")
  }
}

///|
test "noncontiguous synchronous copies lower to typed scalar strides" {
  let program = @luna_tile_ir.admit_luna_tile_program(
    @luna_tile_ir.LunaTileProgramInput::new(
      loop_extents=[1L],
      tensors=[
        generic_cuda_tensor(1, Float32, Global, [2L], [8L], 0L, 4L),
        generic_cuda_tensor(2, Float32, Register, [2L], [8L], 0L, 4L),
      ],
      instructions=[
        LunaTileCopy(generic_cuda_region(1, [2L]), generic_cuda_region(2, [2L])),
      ],
      constraints=[],
    ),
    @luna_tile_ir.LunaTileLimits::new(
      max_tensors=2,
      max_instructions=1,
      max_rank=1,
      max_loop_rank=1,
      max_constraints=0,
      max_canonical_bytes=4096,
      max_shared_memory_bytes=0L,
    ),
  )
  let planning = @luna_tile_ir.plan_luna_tile_cuda_aot_input(
    program,
    @luna_tile_ir.LunaTileAotPlanningPolicy::new(
      vector_bytes=16,
      warp_width=32,
      pipeline_stages=1,
      max_shared_memory_bytes=0L,
    ),
  )
  let lowered = @luna_tile_ir.lower_luna_tile_cuda_translation_unit(
    program,
    planning,
    @luna_tile_ir.LunaTileEntryPointId::from_stable_int(2),
    @luna_tile_ir.LunaTileCudaLoweringLimits::new(max_source_bytes=8192),
  )
  let source = @utf8.decode(lowered.source_bytes())
  assert_true(source.contains("reinterpret_cast<float*>"))
  assert_true(source.contains("c0_0"))
  assert_true(source.contains(" * 8"))
  assert_false(source.contains("reinterpret_cast<uint4*>"))
  assert_false(source.contains("__pipeline_memcpy_async"))
}

///|
fn luna_tile_test_limits() -> @luna_tile_ir.LunaTileLimits raise {
  @luna_tile_ir.LunaTileLimits::new(
    max_tensors=8,
    max_instructions=16,
    max_rank=4,
    max_loop_rank=4,
    max_constraints=8,
    max_canonical_bytes=16384,
    max_shared_memory_bytes=4096L,
  )
}

///|
fn luna_tile_test_tensor(
  id : Int,
  space : @luna_tile_ir.LunaTileAddressSpace,
) -> @luna_tile_ir.LunaTileTensorView raise {
  @luna_tile_ir.LunaTileTensorView::new(
    id=@luna_tile_ir.LunaTileTensorId::from_stable_int(id),
    dtype=BFloat16,
    address_space=space,
    shape=[2L, 2L],
    strides_bytes=[4L, 2L],
    offset_bytes=0L,
    alignment_bytes=4L,
  )
}

///|
fn luna_tile_test_region(id : Int) -> @luna_tile_ir.LunaTileRegion raise {
  @luna_tile_ir.LunaTileRegion::new(
    tensor=@luna_tile_ir.LunaTileTensorId::from_stable_int(id),
    origins=[
      @luna_tile_ir.LunaTileAffineExpr::new(constant=0L, terms=[]),
      @luna_tile_ir.LunaTileAffineExpr::new(constant=0L, terms=[]),
    ],
    extents=[2L, 2L],
  )
}

///|
fn luna_tile_residual_tensor(
  id : Int,
  width : Int64,
  dtype? : @luna_tile_ir.LunaTileDType = BFloat16,
) -> @luna_tile_ir.LunaTileTensorView raise {
  @luna_tile_ir.LunaTileTensorView::new(
    id=@luna_tile_ir.LunaTileTensorId::from_stable_int(id),
    dtype~,
    address_space=Register,
    shape=[width],
    strides_bytes=[
      match dtype {
        BFloat16 | Float16 => 2L
        Float32 | Int32 => 4L
        Int8 => 1L
      },
    ],
    offset_bytes=0L,
    alignment_bytes=2L,
  )
}

///|
fn luna_tile_residual_program(
  kind? : @luna_tile_ir.LunaTileElementwiseKind = Add,
  declared_width? : Int64 = 4L,
  tensor_width? : Int64 = 4L,
  dtype? : @luna_tile_ir.LunaTileDType = BFloat16,
) -> @luna_tile_ir.LunaTileProgram raise {
  let left = @luna_tile_ir.LunaTileTensorId::from_stable_int(1)
  let right = @luna_tile_ir.LunaTileTensorId::from_stable_int(2)
  let output = @luna_tile_ir.LunaTileTensorId::from_stable_int(3)
  @luna_tile_ir.admit_luna_tile_program(
    @luna_tile_ir.LunaTileProgramInput::residual_add(
      left~,
      right~,
      output~,
      width=declared_width,
      loop_extents=[1L],
      tensors=[
        luna_tile_residual_tensor(1, tensor_width, dtype~),
        luna_tile_residual_tensor(2, tensor_width, dtype~),
        luna_tile_residual_tensor(3, tensor_width, dtype~),
      ],
      instructions=[
        @luna_tile_ir.LunaTileInstruction::elementwise(kind, output, [
          left, right,
        ]),
      ],
      constraints=[],
    ),
    luna_tile_test_limits(),
  )
}

///|
fn luna_tile_test_program(
  reverse_constraints? : Bool = false,
) -> @luna_tile_ir.LunaTileProgram raise {
  let first = @luna_tile_ir.LunaTileDimensionEquals(
    @luna_tile_ir.LunaTileConstraintId::from_stable_int(1),
    @luna_tile_ir.LunaTileTensorId::from_stable_int(1),
    0,
    2L,
  )
  let second = @luna_tile_ir.LunaTileAlignmentAtLeast(
    @luna_tile_ir.LunaTileConstraintId::from_stable_int(2),
    @luna_tile_ir.LunaTileTensorId::from_stable_int(2),
    2L,
  )
  let constraints = if reverse_constraints {
    [second, first]
  } else {
    [first, second]
  }
  let stage = @luna_tile_ir.LunaTileStageId::from_stable_int(1)
  @luna_tile_ir.admit_luna_tile_program(
    @luna_tile_ir.LunaTileProgramInput::new(
      loop_extents=[1L],
      tensors=[
        luna_tile_test_tensor(1, Global),
        luna_tile_test_tensor(2, Shared),
      ],
      instructions=[
        LunaTilePipelineStage(stage),
        LunaTileAsyncCopy(
          luna_tile_test_region(1),
          luna_tile_test_region(2),
          stage,
        ),
        LunaTileBarrier(stage, Block),
      ],
      constraints~,
    ),
    luna_tile_test_limits(),
  )
}

///|
test "LunaTile canonicalization and CUDA AOT planning are deterministic" {
  let first = luna_tile_test_program()
  let second = luna_tile_test_program(reverse_constraints=true)
  assert_eq(first.digest(), second.digest())
  assert_eq(first.canonical_bytes(), second.canonical_bytes())
  assert_true(first.residual_add_semantics() is None)
  let policy = @luna_tile_ir.LunaTileAotPlanningPolicy::new(
    vector_bytes=4,
    warp_width=32,
    pipeline_stages=2,
    max_shared_memory_bytes=4096L,
  )
  let planned = @luna_tile_ir.plan_luna_tile_cuda_aot_input(first, policy)
  let repeated = @luna_tile_ir.plan_luna_tile_cuda_aot_input(second, policy)
  assert_eq(planned.digest(), repeated.digest())
  assert_eq(planned.memory_plan().shared_tensor_count(), 1)
  assert_true(planned.memory_plan().shared_memory_bytes() > 0L)
}

///|
test "exact residual-add semantics differ from a neutral generic graph" {
  let residual = luna_tile_residual_program()
  guard residual.residual_add_semantics() is Some(semantic) else {
    fail("exact residual semantics were not published")
  }
  assert_eq(semantic.width(), 4L)
  let wider = luna_tile_residual_program(declared_width=5L, tensor_width=5L)
  assert_true(residual.digest() != wider.digest())
  let left = @luna_tile_ir.LunaTileTensorId::from_stable_int(1)
  let right = @luna_tile_ir.LunaTileTensorId::from_stable_int(2)
  let output = @luna_tile_ir.LunaTileTensorId::from_stable_int(3)
  let neutral = @luna_tile_ir.admit_luna_tile_program(
    @luna_tile_ir.LunaTileProgramInput::new(
      loop_extents=[1L],
      tensors=[
        luna_tile_residual_tensor(1, 4L),
        luna_tile_residual_tensor(2, 4L),
        luna_tile_residual_tensor(3, 4L),
      ],
      instructions=[
        @luna_tile_ir.LunaTileInstruction::elementwise(Add, output, [
          left, right,
        ]),
      ],
      constraints=[],
    ),
    luna_tile_test_limits(),
  )
  assert_true(neutral.residual_add_semantics() is None)
  assert_true(neutral.digest() != residual.digest())
}

///|
test "residual semantic admission rejects algorithm and shape substitution" {
  try luna_tile_residual_program(kind=Multiply) catch {
    @luna_tile_ir.InvalidProgram(SemanticContract) => ()
    _ => fail("unexpected residual algorithm failure")
  } noraise {
    _ => fail("multiply graph asserted residual-add semantics")
  }
  try luna_tile_residual_program(declared_width=3L) catch {
    @luna_tile_ir.InvalidProgram(SemanticContract) => ()
    _ => fail("unexpected residual width failure")
  } noraise {
    _ => fail("wrong-width graph asserted residual-add semantics")
  }
  try luna_tile_residual_program(dtype=Float16) catch {
    @luna_tile_ir.InvalidProgram(SemanticContract) => ()
    _ => fail("unexpected residual dtype failure")
  } noraise {
    _ => fail("non-BF16 graph asserted residual-add semantics")
  }
  let first = @luna_tile_ir.LunaTileTensorId::from_stable_int(1)
  let second = @luna_tile_ir.LunaTileTensorId::from_stable_int(2)
  try
    @luna_tile_ir.admit_luna_tile_program(
      @luna_tile_ir.LunaTileProgramInput::residual_add(
        left=first,
        right=second,
        output=first,
        width=4L,
        loop_extents=[1L],
        tensors=[
          luna_tile_residual_tensor(1, 4L),
          luna_tile_residual_tensor(2, 4L),
          luna_tile_residual_tensor(3, 4L),
        ],
        instructions=[
          @luna_tile_ir.LunaTileInstruction::elementwise(Add, first, [
            first, second,
          ]),
        ],
        constraints=[],
      ),
      luna_tile_test_limits(),
    )
  catch {
    @luna_tile_ir.InvalidProgram(SemanticContract) => ()
    _ => fail("unexpected residual output-role failure")
  } noraise {
    _ => fail("residual output aliased an input role")
  }
}

///|
test "LunaTile async copy requires an exact closing barrier" {
  let stage = @luna_tile_ir.LunaTileStageId::from_stable_int(1)
  try
    @luna_tile_ir.admit_luna_tile_program(
      @luna_tile_ir.LunaTileProgramInput::new(
        loop_extents=[1L],
        tensors=[
          luna_tile_test_tensor(1, Global),
          luna_tile_test_tensor(2, Shared),
        ],
        instructions=[
          LunaTilePipelineStage(stage),
          LunaTileAsyncCopy(
            luna_tile_test_region(1),
            luna_tile_test_region(2),
            stage,
          ),
        ],
        constraints=[],
      ),
      luna_tile_test_limits(),
    )
  catch {
    @luna_tile_ir.InvalidProgram(PipelineBarrier) => ()
    _ => fail("unexpected LunaTile validation failure")
  } noraise {
    _ => fail("missing async-copy barrier was admitted")
  }
}

///|
test "LunaTile admission enforces the shared-memory envelope" {
  let stage = @luna_tile_ir.LunaTileStageId::from_stable_int(1)
  try
    @luna_tile_ir.admit_luna_tile_program(
      @luna_tile_ir.LunaTileProgramInput::new(
        loop_extents=[1L],
        tensors=[
          luna_tile_test_tensor(1, Global),
          luna_tile_test_tensor(2, Shared),
        ],
        instructions=[
          LunaTilePipelineStage(stage),
          LunaTileAsyncCopy(
            luna_tile_test_region(1),
            luna_tile_test_region(2),
            stage,
          ),
          LunaTileBarrier(stage, Block),
        ],
        constraints=[],
      ),
      @luna_tile_ir.LunaTileLimits::new(
        max_tensors=2,
        max_instructions=3,
        max_rank=2,
        max_loop_rank=1,
        max_constraints=0,
        max_canonical_bytes=4096,
        max_shared_memory_bytes=0L,
      ),
    )
  catch {
    @luna_tile_ir.LimitExceeded(SharedMemoryBytes) => ()
    _ => fail("unexpected shared-memory validation failure")
  } noraise {
    _ => fail("shared-memory envelope was ignored")
  }
}

///|
test "LunaTile compute instructions reject implicit global-memory execution" {
  let first = @luna_tile_ir.LunaTileTensorId::from_stable_int(1)
  let second = @luna_tile_ir.LunaTileTensorId::from_stable_int(2)
  let output = @luna_tile_ir.LunaTileTensorId::from_stable_int(3)
  try
    @luna_tile_ir.admit_luna_tile_program(
      @luna_tile_ir.LunaTileProgramInput::new(
        loop_extents=[1L],
        tensors=[
          luna_tile_test_tensor(1, Global),
          luna_tile_test_tensor(2, Global),
          luna_tile_test_tensor(3, Global),
        ],
        instructions=[
          @luna_tile_ir.LunaTileInstruction::elementwise(Add, output, [
            first, second,
          ]),
        ],
        constraints=[],
      ),
      luna_tile_test_limits(),
    )
  catch {
    @luna_tile_ir.InvalidProgram(OperationAddressSpace) => ()
    _ => fail("unexpected compute-address-space validation failure")
  } noraise {
    _ => fail("implicit global-memory computation was admitted")
  }
}

///|
test "LunaTile shared tensors require an explicit nonoverlapping plan" {
  let stage = @luna_tile_ir.LunaTileStageId::from_stable_int(1)
  try
    @luna_tile_ir.admit_luna_tile_program(
      @luna_tile_ir.LunaTileProgramInput::new(
        loop_extents=[1L],
        tensors=[
          luna_tile_test_tensor(1, Global),
          luna_tile_test_tensor(2, Shared),
          luna_tile_test_tensor(3, Shared),
        ],
        instructions=[
          LunaTilePipelineStage(stage),
          LunaTileAsyncCopy(
            luna_tile_test_region(1),
            luna_tile_test_region(2),
            stage,
          ),
          LunaTileBarrier(stage, Block),
        ],
        constraints=[],
      ),
      luna_tile_test_limits(),
    )
  catch {
    @luna_tile_ir.InvalidTensor(_, TensorOverlap) => ()
    _ => fail("unexpected shared-memory overlap failure")
  } noraise {
    _ => fail("overlapping shared-memory tensors were admitted")
  }
}

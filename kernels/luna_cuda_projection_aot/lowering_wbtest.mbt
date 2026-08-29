///|
test "projection-family candidates are deterministic and non-circular" {
  let families : Array[(Int, ProjectionAotFamily)] = [
    (1, QkvProjection),
    (4, DenseProjection),
    (5, GatedMlp),
    (6, LanguageModelHead),
  ]
  for item in families {
    let (operation, family) = item
    let first = projection_fixture_lowering(operation)
    let second = projection_fixture_lowering(operation)
    assert_eq(first, second)
    assert_eq(first.family(), family)
    assert_eq(first.operation_id().as_int(), operation)
    assert_eq(
      first.entry_point_id(),
      projection_fixture_entry(operation).entry_point_id(),
    )
    assert_eq(
      first.launch_dimensions(),
      projection_fixture_dimensions(operation),
    )
    assert_eq(
      first.operands(),
      ReadOnlyArray::from_array(projection_fixture_operands(operation)),
    )
    assert_eq(first.target(), projection_fixture_target())
    assert_eq(first.profile().max_query_tokens(), 4)
    assert_eq(first.launch_dimensions().block_x(), 256)
  }
}

///|
test "generated projection sources retain exact canonical ABIs" {
  let qkv = @utf8.decode(projection_fixture_lowering(1).source_bytes())
  assert_true(
    qkv.contains(
      "const int32_t *counts, const __nv_bfloat16 *input, const __nv_bfloat16 *query_weight, const __nv_bfloat16 *key_weight, const __nv_bfloat16 *value_weight, __nv_bfloat16 *output",
    ),
  )
  assert_true(qkv.contains("const int token_count = counts[3];"))
  assert_true(qkv.contains("weight = value_weight"))
  assert_true(qkv.contains("accumulator = __fadd_rn(accumulator, product);"))
  assert_false(qkv.contains("malloc"))
  assert_false(qkv.contains("cublas"))
  let dense = @utf8.decode(projection_fixture_lowering(4).source_bytes())
  assert_true(dense.contains("const __nv_bfloat16 *weight"))
  assert_true(dense.contains("__float2bfloat16_rn(accumulator)"))
  let mlp = @utf8.decode(projection_fixture_lowering(5).source_bytes())
  assert_true(mlp.contains("const __nv_bfloat16 *gate_weight"))
  assert_true(mlp.contains("const __nv_bfloat16 *up_weight"))
  assert_true(mlp.contains("const __nv_bfloat16 *down_weight"))
  assert_true(mlp.contains("expf(-gate)"))
  assert_true(mlp.contains("__float2bfloat16_rn(output_accumulator)"))
}

///|
test "Qwen-shaped QKV split is authenticated from all three weight bytes" {
  let operation = projection_fixture_operation(
    7,
    QkvProjection,
    [
      Activation(@plan.ActivationRef::new(0)),
      Tensor(@plan.TensorRef::new(9)),
      Tensor(@plan.TensorRef::new(10)),
      Tensor(@plan.TensorRef::new(11)),
    ],
    ProjectionShape(input_width=4, output_width=16),
    @plan.KernelCapabilityId::qkv_projection(),
  )
  let operands = [
    projection_fixture_operand(RuntimeInput(StepCounts), 20L, 4L),
    projection_fixture_operand(
      ActivationInput(@plan.ActivationRef::new(0)),
      32L,
      16L,
    ),
    projection_fixture_operand(WeightInput(@plan.TensorRef::new(9)), 64L, 16L),
    projection_fixture_operand(WeightInput(@plan.TensorRef::new(10)), 32L, 16L),
    projection_fixture_operand(WeightInput(@plan.TensorRef::new(11)), 32L, 16L),
    projection_fixture_operand(
      ActivationOutput(@plan.ActivationRef::new(7)),
      128L,
      16L,
    ),
  ]
  let contract = projection_fixture_contracts().contracts()[1]
  let lowering = lower_projection_cuda_aot_candidate(
    operation,
    contract.profile(),
    @catalog.AotKernelEntryPointId::from_stable_int(8),
    projection_fixture_target(),
    projection_fixture_compiler(),
    operands,
    projection_fixture_limits(),
  )
  assert_eq(lowering.input_width(), 4)
  assert_eq(lowering.output_width(), 16)
  assert_eq(lowering.intermediate_width(), 4)
  let source = @utf8.decode(lowering.source_bytes())
  assert_true(source.contains("if (column >= 8)"))
  assert_true(source.contains("local_column = column - 8"))
  let mismatched = operands.copy()
  mismatched[4] = projection_fixture_operand(
    WeightInput(@plan.TensorRef::new(11)),
    16L,
    16L,
  )
  try
    lower_projection_cuda_aot_candidate(
      operation,
      contract.profile(),
      @catalog.AotKernelEntryPointId::from_stable_int(8),
      projection_fixture_target(),
      projection_fixture_compiler(),
      mismatched,
      projection_fixture_limits(),
    )
  catch {
    Mismatch(OperandSize) => ()
    _ => fail("unexpected asymmetric K/V rejection")
  } noraise {
    _ => fail("expected exact Q/K/V split rejection")
  }
}

///|
test "candidate recipe excludes final module identity" {
  let lowering = projection_fixture_lowering(1)
  let recipe = @utf8.decode(lowering.recipe_bytes())
  assert_true(
    recipe.contains("schema=lunaflux-luna-cuda-projection-aot-recipe-v1\n"),
  )
  assert_true(recipe.contains("family=qkv_projection\n"))
  assert_true(recipe.contains("profile_id=1\n"))
  assert_false(recipe.contains("module_sha256="))
  assert_false(recipe.contains("family_id="))
  assert_true(recipe.contains("entry_point_id=2\n"))
  assert_true(recipe.contains("grid=1,1,1\nblock=256,1,1\n"))
  assert_true(recipe.contains("manifest_bindable=false\n"))
  assert_true(
    recipe.contains(
      "operands=runtime:step_counts@20/4,activation_in:0@32/16,weight:1@32/16,weight:2@16/16,weight:3@16/16,activation_out:1@64/16\n",
    ),
  )
}

///|
test "strict binder rejects final-contract geometry drift" {
  let plan = projection_fixture_plan()
  let candidate = projection_fixture_lowering(1)
  try
    bind_manifest_projection_cuda_aot(
      candidate,
      plan.operations()[1],
      projection_fixture_contracts(qkv_grid_x=2),
    )
  catch {
    Mismatch(LaunchDimensions) => ()
    _ => fail("unexpected dimension rejection")
  } noraise {
    _ => fail("expected dimension rejection")
  }
  try {
    let contract = projection_fixture_contracts(qkv_query_weight_bytes=34L).contracts()[1]
    lower_projection_cuda_aot_candidate(
      plan.operations()[1],
      contract.profile(),
      contract.entry_point().entry_point_id(),
      projection_fixture_target(),
      projection_fixture_compiler(),
      contract.operands(),
      projection_fixture_limits(),
    )
  } catch {
    Mismatch(OperandSize) => ()
    _ => fail("unexpected operand-size rejection")
  } noraise {
    _ => fail("expected operand-size rejection")
  }
}

///|
test "lowering rejects foreign operation target compiler and source limit" {
  let plan = projection_fixture_plan()
  let contract = projection_fixture_contracts().contracts()[1]
  try
    lower_projection_cuda_aot_candidate(
      plan.operations()[4],
      contract.profile(),
      contract.entry_point().entry_point_id(),
      projection_fixture_target(),
      projection_fixture_compiler(),
      contract.operands(),
      projection_fixture_limits(),
    )
  catch {
    Mismatch(_) => ()
    _ => fail("unexpected operation rejection")
  } noraise {
    _ => fail("expected operation rejection")
  }
  try
    lower_projection_cuda_aot_candidate(
      plan.operations()[1],
      contract.profile(),
      contract.entry_point().entry_point_id(),
      projection_fixture_target(bf16=false),
      projection_fixture_compiler(),
      contract.operands(),
      projection_fixture_limits(),
    )
  catch {
    Unsupported(DeviceTarget) => ()
    _ => fail("unexpected target rejection")
  } noraise {
    _ => fail("expected target rejection")
  }
  try
    lower_projection_cuda_aot_candidate(
      plan.operations()[1],
      contract.profile(),
      contract.entry_point().entry_point_id(),
      projection_fixture_target(),
      projection_fixture_compiler(reassociate=true),
      contract.operands(),
      projection_fixture_limits(),
    )
  catch {
    Unsupported(CompilerPolicy) => ()
    _ => fail("unexpected compiler rejection")
  } noraise {
    _ => fail("expected compiler rejection")
  }
  try
    lower_projection_cuda_aot_candidate(
      plan.operations()[1],
      contract.profile(),
      contract.entry_point().entry_point_id(),
      projection_fixture_target(),
      projection_fixture_compiler(),
      contract.operands(),
      projection_fixture_limits(max_source_bytes=32),
    )
  catch {
    LimitExceeded(Source) => ()
    _ => fail("unexpected source-limit rejection")
  } noraise {
    _ => fail("expected source-limit rejection")
  }
}

///|
test "strict binder authenticates the final module only after compilation" {
  let plan = projection_fixture_plan()
  let contracts = projection_fixture_contracts()
  let candidate = projection_fixture_lowering(1)
  let bound = bind_manifest_projection_cuda_aot(
    candidate,
    plan.operations()[1],
    contracts,
  )
  assert_true(bound.manifest_bindable())
  assert_eq(bound.candidate(), candidate)
  assert_eq(bound.entry_point(), projection_fixture_entry(1))
  let record = @utf8.decode(bound.binding_record_bytes())
  assert_true(record.contains("module_sha256=" + "2".repeat(64)))
  assert_true(record.contains("manifest_bindable=true\n"))
}

///|
test "projection content address rejects non-canonical digest" {
  assert_eq(
    projection_content_addressed_cubin_path("a".repeat(64)),
    "sha256/" + "a".repeat(64) + ".cubin",
  )
  try projection_content_addressed_cubin_path("ABC") catch {
    Invalid(Digest) => ()
    _ => fail("unexpected digest rejection")
  } noraise {
    _ => fail("expected digest rejection")
  }
}

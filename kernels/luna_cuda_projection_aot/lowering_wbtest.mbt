///|
test "CUDA lowering consumes the startup-selected backend-neutral strategy" {
  let operation = projection_fixture_operation(
    11,
    OutputProjection,
    [Activation(@plan.ActivationRef::new(9)), Tensor(@plan.TensorRef::new(15))],
    ProjectionShape(input_width=1024, output_width=1024),
    @plan.KernelCapabilityId::dense_projection(),
  )
  let operands = [
    projection_fixture_operand(RuntimeInput(StepCounts), 20L, 4L),
    projection_fixture_operand(
      ActivationInput(@plan.ActivationRef::new(9)),
      8192L,
      16L,
    ),
    projection_fixture_operand(
      WeightInput(@plan.TensorRef::new(15)),
      2097152L,
      16L,
    ),
    projection_fixture_operand(
      ActivationOutput(@plan.ActivationRef::new(11)),
      8192L,
      16L,
    ),
  ]
  let target = projection_fixture_target()
  let profile = projection_fixture_contracts().contracts()[5].profile()
  let shape = @luna_projection_strategy.ProjectionShape::new(
    Dense,
    input_width=1024,
    output_width=1024,
  )
  let strategy = @luna_projection_strategy.ProjectionStrategy::new(
    stable_id=700,
    phase=Prefill,
    strategy_class=Some(SmallRowTiledGemm),
    kernel_class=ScalarTile,
    row_bucket=8,
    tile=@luna_projection_strategy.ProjectionTile::new(
      rows=1,
      columns=32,
      reduction=1,
    ),
  )
  let decode_strategy = @luna_projection_strategy.ProjectionStrategy::new(
    stable_id=699,
    phase=Decode,
    strategy_class=Some(DecodeGemv),
    kernel_class=ScalarTile,
    row_bucket=1,
    tile=@luna_projection_strategy.ProjectionTile::new(
      rows=1,
      columns=32,
      reduction=32,
    ),
  )
  let prefill_record = @luna_projection_strategy.ProjectionAutotuneRecord::new(
    key=@luna_projection_strategy.ProjectionAutotuneKey::new(
      backend=@luna_projection_strategy.ProjectionBackendId::new("cuda-aot-v1"),
      target=@luna_projection_strategy.ProjectionTargetId::new("sm_90"),
      numeric_type=ProjectionBFloat16,
      shape~,
      strategy~,
    ),
    median_latency_ns=90L,
    sample_count=9,
  )
  let decode_record = @luna_projection_strategy.ProjectionAutotuneRecord::new(
    key=@luna_projection_strategy.ProjectionAutotuneKey::new(
      backend=@luna_projection_strategy.ProjectionBackendId::new("cuda-aot-v1"),
      target=@luna_projection_strategy.ProjectionTargetId::new("sm_90"),
      numeric_type=ProjectionBFloat16,
      shape~,
      strategy=decode_strategy,
    ),
    median_latency_ns=80L,
    sample_count=9,
  )
  let runtime = prepare_projection_cuda_runtime_dispatch(
    operation,
    profile,
    target,
    operands,
    [prefill_record, decode_record],
  )
  assert_eq(runtime.select(4).source(), OfflineAutotune)
  let lowering = lower_projection_cuda_aot_candidate_with_dispatch(
    operation,
    profile,
    @catalog.AotKernelEntryPointId::from_stable_int(12),
    target,
    projection_fixture_compiler(),
    operands,
    runtime,
    projection_fixture_limits(),
  )
  assert_eq(lowering.dispatch_plan().prefill().stable_id(), 700)
  let source = @utf8.decode(lowering.source_bytes())
  assert_false(source.contains("nvcuda::wmma::mma_sync"))
  assert_true(
    source.contains("accumulator = __fadd_rn(accumulator, __fmul_rn("),
  )
  let recipe = @utf8.decode(lowering.recipe_bytes())
  assert_true(recipe.contains("prefill_strategy=scalar-tile\n"))
  assert_true(recipe.contains("prefill_strategy_class=small-row-tiled-gemm\n"))
  let fallback = lower_projection_cuda_aot_candidate(
    operation,
    profile,
    @catalog.AotKernelEntryPointId::from_stable_int(12),
    target,
    projection_fixture_compiler(),
    operands,
    projection_fixture_limits(),
  )
  assert_true(
    @utf8.decode(fallback.source_bytes()).contains("mma.sync.aligned.m16n8k16"),
  )
  assert_false(
    @utf8.decode(fallback.source_bytes()).contains("__nv_bfloat16 *input_tile"),
  )
  assert_false(
    @utf8.decode(fallback.source_bytes()).contains("matrix_output_bytes"),
  )
  assert_eq(
    fallback.tile_compilation().schedule().row_tail_policy(),
    ZeroPaddedMatrixRowTail,
  )
  assert_eq(fallback.launch_dimensions().shared_memory_bytes(), 8192)
  let compiler_source = projection_source(
    "projection_compiler",
    projection_parameters(operation),
    profile.max_query_tokens(),
    profile.max_query_rows(),
    fallback.dispatch_plan(),
    fallback.tile_compilation(),
    false,
    false,
    max_registers=128,
  )
  assert_false(
    @utf8.decode(compiler_source).contains("__nv_bfloat16 *input_tile"),
  )
  assert_false(@utf8.decode(compiler_source).contains("wmma::"))
  let foreign_target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
  try
    lower_projection_cuda_aot_candidate_with_dispatch(
      operation,
      profile,
      @catalog.AotKernelEntryPointId::from_stable_int(12),
      foreign_target,
      projection_fixture_compiler(),
      operands,
      runtime,
      projection_fixture_limits(),
    )
  catch {
    Mismatch(Strategy) => ()
    _ => fail("unexpected foreign projection strategy rejection")
  } noraise {
    _ => fail("expected target-bound projection strategy rejection")
  }
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
  assert_true(source.contains("if (local_column >= 8)"))
  assert_true(source.contains("local_column -= 8"))
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
  assert_true(recipe.contains("shared_memory_bytes=0\n"))
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

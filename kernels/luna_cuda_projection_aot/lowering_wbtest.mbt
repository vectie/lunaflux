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
  assert_true(mlp.contains("extern __shared__ float combined[];"))
  assert_true(mlp.contains("const int token = (int)blockIdx.x;"))
  assert_true(mlp.contains("__syncthreads();"))
  assert_true(mlp.contains("__float2bfloat16_rn(output_accumulator)"))
  assert_eq(
    projection_fixture_lowering(5).launch_dimensions().shared_memory_bytes(),
    32,
  )
  assert_eq(projection_fixture_lowering(5).launch_dimensions().grid_x(), 4)
}

///|
test "Qwen-shaped MLP tiles prefill and uses warp GEMV for decode" {
  let operation = projection_fixture_operation(
    9,
    GatedMlp,
    [
      Activation(@plan.ActivationRef::new(8)),
      Tensor(@plan.TensorRef::new(12)),
      Tensor(@plan.TensorRef::new(13)),
      Tensor(@plan.TensorRef::new(14)),
    ],
    GatedMlpShape(hidden_size=1024, intermediate_size=3072),
    @plan.KernelCapabilityId::gated_mlp(),
  )
  let operands = [
    projection_fixture_operand(RuntimeInput(StepCounts), 20L, 4L),
    projection_fixture_operand(
      ActivationInput(@plan.ActivationRef::new(8)),
      8192L,
      16L,
    ),
    projection_fixture_operand(
      WeightInput(@plan.TensorRef::new(12)),
      6291456L,
      16L,
    ),
    projection_fixture_operand(
      WeightInput(@plan.TensorRef::new(13)),
      6291456L,
      16L,
    ),
    projection_fixture_operand(
      WeightInput(@plan.TensorRef::new(14)),
      6291456L,
      16L,
    ),
    projection_fixture_operand(
      ActivationOutput(@plan.ActivationRef::new(9)),
      8192L,
      16L,
    ),
    projection_fixture_operand(Workspace, 24576L, 16L),
  ]
  let profile = projection_fixture_contracts().contracts()[5].profile()
  let lowering = lower_projection_cuda_aot_candidate(
    operation,
    profile,
    @catalog.AotKernelEntryPointId::from_stable_int(10),
    projection_fixture_target(),
    projection_fixture_compiler(),
    operands,
    projection_fixture_limits(),
  )
  let source = @utf8.decode(lowering.source_bytes())
  assert_true(source.contains("__nv_bfloat16 *workspace"))
  assert_true(source.contains("_down(const int32_t *counts"))
  assert_true(source.contains("lf_projection_warp_dot2(input"))
  assert_true(source.contains("lf_projection_warp_dot(workspace"))
  assert_true(source.contains("__bfloat1622float2"))
  assert_true(
    source.contains("workspace[(unsigned long long)(row_start + tile_row)"),
  )
  assert_true(source.contains("* 16;"))
  assert_true(source.contains("float gate_values[8];"))
  assert_true(
    source.contains(
      "load_matrix_sync(a, input + (unsigned long long)row_start * 1024ULL",
    ),
  )
  assert_true(
    source.contains(
      "load_matrix_sync(a, workspace + (unsigned long long)row_start * 3072ULL",
    ),
  )
  assert_false(source.contains("__nv_bfloat16 *a_tile"))
  assert_true(source.contains("row_start + tile_row < token_count"))
  assert_true(source.contains("if (token_count == 1)"))
  assert_true(source.contains("__shfl_down_sync"))
  assert_true(source.contains("if (warp >= 4) return;"))
  assert_true(source.contains("const int group_count = 16;"))
  assert_true(source.contains("work_group += (int)gridDim.x"))
  assert_true(source.contains("group_index * 64 + warp * 16"))
  assert_eq(lowering.launch_dimensions().grid_x(), 12)
  assert_eq(lowering.launch_dimensions().block_x(), 512)
  assert_eq(lowering.launch_dimensions().shared_memory_bytes(), 16384)
  assert_true(@utf8.decode(lowering.recipe_bytes()).contains("block=512,1,1\n"))
}

///|
test "Qwen-shaped language head uses tiles for prefill and GEMV for sampling" {
  let operation = projection_fixture_operation(
    10,
    LanguageModelHead,
    [Activation(@plan.ActivationRef::new(9)), Tensor(@plan.TensorRef::new(0))],
    ProjectionShape(input_width=1024, output_width=151936),
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
      WeightInput(@plan.TensorRef::new(0)),
      311164928L,
      16L,
    ),
    projection_fixture_operand(
      ActivationOutput(@plan.ActivationRef::new(10)),
      1215488L,
      16L,
    ),
  ]
  let profile = projection_fixture_contracts().contracts()[5].profile()
  let lowering = lower_projection_cuda_aot_candidate(
    operation,
    profile,
    @catalog.AotKernelEntryPointId::from_stable_int(11),
    projection_fixture_target(),
    projection_fixture_compiler(),
    operands,
    projection_fixture_limits(),
  )
  let source = @utf8.decode(lowering.source_bytes())
  assert_true(source.contains("if (counts[2] == 1)"))
  assert_true(source.contains("const unsigned long long live_row"))
  assert_true(source.contains("nvcuda::wmma::mma_sync"))
  assert_true(source.contains("__shfl_down_sync"))
  assert_eq(lowering.launch_dimensions().grid_x(), 1187)
}

///|
test "Qwen tensor-core QKV and dense use warp GEMV for one-token decode" {
  let qkv_parameters : ProjectionParameters = {
    family: QkvProjection,
    input_width: 1024,
    query_width: 2048,
    output_width: 4096,
    intermediate_width: 1024,
  }
  let qkv = @utf8.decode(
    projection_tensor_core_qkv_source("qwen_qkv", qkv_parameters, 256),
  )
  assert_true(qkv.contains("if (token_count == 1)"))
  assert_true(qkv.contains("lf_projection_warp_dot(input, selected_weight"))
  assert_true(qkv.contains("__bfloat1622float2"))
  assert_true(qkv.contains("__shfl_down_sync"))
  assert_true(qkv.contains("nvcuda::wmma::mma_sync"))
  assert_true(qkv.contains("const int group_count = 32;"))
  assert_true(qkv.contains("blockIdx.x / group_count) * 16"))
  let dense_parameters : ProjectionParameters = {
    family: DenseProjection,
    input_width: 1024,
    query_width: 0,
    output_width: 1024,
    intermediate_width: 0,
  }
  let dense = @utf8.decode(
    projection_tensor_core_dense_source("qwen_dense", dense_parameters, 256),
  )
  assert_true(dense.contains("if (token_count == 1)"))
  assert_true(dense.contains("lf_projection_warp_dot(input, weight"))
  assert_true(dense.contains("__bfloat1622float2"))
  assert_true(dense.contains("__shfl_down_sync"))
  assert_true(dense.contains("nvcuda::wmma::mma_sync"))
  assert_true(dense.contains("const int group_count = 8;"))
  let profile = projection_fixture_contracts().contracts()[1].profile()
  let records : Array[@luna_projection_strategy.ProjectionAutotuneRecord] = []
  let qkv_dispatch = projection_build_runtime_dispatch(
    qkv_parameters,
    profile.max_query_tokens(),
    projection_fixture_target(),
    records,
  ).dispatch_plan(profile.max_query_tokens())
  let dense_dispatch = projection_build_runtime_dispatch(
    dense_parameters,
    profile.max_query_tokens(),
    projection_fixture_target(),
    records,
  ).dispatch_plan(profile.max_query_tokens())
  assert_eq(
    projection_dimensions(profile, qkv_parameters, qkv_dispatch).grid_x(),
    512,
  )
  assert_eq(
    projection_dimensions(profile, dense_parameters, dense_dispatch).grid_x(),
    128,
  )
}

///|
test "single-row profiles reuse the exact decode projection strategy" {
  let target = projection_fixture_target()
  let records : Array[@luna_projection_strategy.ProjectionAutotuneRecord] = []
  let scalar_parameters : ProjectionParameters = {
    family: DenseProjection,
    input_width: 4,
    query_width: 0,
    output_width: 8,
    intermediate_width: 0,
  }
  let scalar_runtime = projection_build_runtime_dispatch(
    scalar_parameters, 1, target, records,
  )
  let scalar = projection_validate_runtime_dispatch(
    scalar_runtime, scalar_parameters, 1, target,
  )
  assert_eq(scalar.decode(), scalar.prefill())
  assert_eq(scalar.decode().kernel_class(), ScalarTile)
  let subgroup_parameters : ProjectionParameters = {
    family: DenseProjection,
    input_width: 32,
    query_width: 0,
    output_width: 64,
    intermediate_width: 0,
  }
  let subgroup_runtime = projection_build_runtime_dispatch(
    subgroup_parameters, 1, target, records,
  )
  let subgroup = projection_validate_runtime_dispatch(
    subgroup_runtime, subgroup_parameters, 1, target,
  )
  assert_eq(subgroup.decode(), subgroup.prefill())
  assert_eq(subgroup.decode().kernel_class(), SubgroupGemv)
}

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
  assert_true(source.contains("accumulator = __fadd_rn(accumulator, product)"))
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
    @utf8.decode(fallback.source_bytes()).contains("nvcuda::wmma::mma_sync"),
  )
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

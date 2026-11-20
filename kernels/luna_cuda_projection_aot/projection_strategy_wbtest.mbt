///|
test "projection-family candidates are deterministic and non-circular" {
  let families : Array[(Int32, ProjectionAotFamily)] = [
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
    let expected_demand : @luna_projection_tile_compiler.ProjectionResultDemand = if family ==
      LanguageModelHead {
      QueryRowEnds
    } else {
      AllTokenRows
    }
    assert_eq(first.tile_compilation().schedule().demand(), expected_demand)
    assert_eq(
      first.tile_compilation().schedule().single_row_reduction(),
      OrderedDot,
    )
    assert_eq(first.tile_compilation().digest().as_hex().length(), 64)
    assert_true(
      @utf8.decode(first.recipe_bytes()).contains(
        "functional_compilation_sha256=" +
        first.tile_compilation().digest().as_hex() +
        "\n",
      ),
    )
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
  assert_true(qkv.contains("accumulator = __fadd_rn(accumulator, __fmul_rn("))
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
  assert_true(mlp.contains("const int token = blockIdx.x;"))
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
  assert_true(source.contains("workspace[(unsigned long long)output_row"))
  assert_true(source.contains("* 16;"))
  // This bounded fixture has one matrix row tile, so its two sibling folds
  // keep the epilogue in registers instead of materializing padded reuse rows.
  assert_eq(
    lowering
    .tile_compilation()
    .schedule()
    .sibling_pipeline()
    .unwrap()
    .row_width(),
    16,
  )
  assert_true(source.contains("float cg[1][8] = {}, cu[1][8] = {};"))
  assert_true(
    source.contains("const float gate = cg[row][item], up = cu[row][item];"),
  )
  assert_false(source.contains("float *gate_tiles"))
  assert_false(source.contains("__nv_bfloat16 *input_tile"))
  assert_true(source.contains("ldmatrix.sync.aligned.m8n8.x4.shared.b16"))
  assert_true(
    source.contains("workspace[(unsigned long long)output_row * 3072ULL"),
  )
  assert_false(source.contains("__nv_bfloat16 *a_tile"))
  assert_true(source.contains("output_row < token_count"))
  assert_true(source.contains("if (token_count == 1)"))
  assert_true(source.contains("__shfl_down_sync"))
  assert_eq(
    lowering.tile_compilation().schedule().single_row_reduction(),
    StridedPairwiseDot(32),
  )
  assert_true(source.contains("if (blockDim.x != 128) return"))
  assert_true(source.contains("const int group_count = 16;"))
  assert_true(source.contains("work_group += (int)gridDim.x"))
  assert_true(source.contains("group_index * 64 + warp * 16"))
  // The same entry point must also cover single-token GEMV. Prefill's
  // twelve matrix tiles alone do not cover its 192 decode output groups.
  assert_eq(lowering.launch_dimensions().grid_x(), 192)
  assert_eq(lowering.launch_dimensions().block_x(), 512)
  assert_eq(lowering.launch_dimensions().shared_memory_bytes(), 16384)
  assert_true(@utf8.decode(lowering.recipe_bytes()).contains("block=512,1,1\n"))
  for max_registers in [32, 63, 64, 128] {
    let bounded = lower_projection_cuda_aot_candidate(
      operation,
      profile,
      @catalog.AotKernelEntryPointId::from_stable_int(10),
      projection_fixture_target(),
      projection_fixture_compiler(max_registers~),
      operands,
      projection_fixture_limits(),
    )
    let bounded_source = @utf8.decode(bounded.source_bytes())
    assert_false(bounded_source.contains("__launch_bounds__"))
    assert_true(
      bounded_source.contains(
        "extern \"C\" __global__ void " + bounded.function_symbol() + "_down(",
      ),
    )
    assert_eq(bounded.launch_dimensions(), lowering.launch_dimensions())
  }
}

///|
test "Qwen-shaped language head uses tiles for prefill and GEMV for sampling" {
  let operation = projection_fixture_operation(
    10,
    LanguageModelHead,
    [Activation(@plan.ActivationRef::new(9)), Tensor(@plan.TensorRef::new(0))],
    ProjectionShape(input_width=1024, output_width=151936),
    @plan.KernelCapabilityId::dense_projection(),
    runtime_inputs=[StepCounts, QueryRowOffsets],
  )
  let operands = [
    projection_fixture_operand(RuntimeInput(StepCounts), 20L, 4L),
    projection_fixture_operand(RuntimeInput(QueryRowOffsets), 12L, 4L),
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
  assert_true(source.contains("const int selected_row = row_offsets[1] - 1"))
  assert_true(source.contains("const unsigned long long live_row"))
  assert_true(
    source.contains(
      "logical_row < pipeline_rows ? row_offsets[logical_row + 1] - 1",
    ),
  )
  assert_true(source.contains("operand_index(local_row, k)"))
  assert_false(source.contains("wmma::"))
  assert_true(source.contains("__shfl_down_sync"))
  assert_eq(lowering.launch_dimensions().grid_x(), 1187)
  assert_eq(lowering.launch_dimensions().shared_memory_bytes(), 8704)
}

///|
test "unified QKV and dense folds retain one-token reduction and launch ABI" {
  let qkv_parameters : ProjectionParameters = {
    family: QkvProjection,
    input_width: 1024,
    query_width: 2048,
    output_width: 4096,
    intermediate_width: 1024,
  }
  let dense_parameters : ProjectionParameters = {
    family: DenseProjection,
    input_width: 1024,
    query_width: 0,
    output_width: 1024,
    intermediate_width: 0,
  }
  let profile = projection_fixture_contracts().contracts()[1].profile()
  let records : Array[@luna_projection_strategy.ProjectionAutotuneRecord] = []
  let qkv_dispatch = projection_build_runtime_dispatch(
    qkv_parameters,
    1024,
    projection_fixture_target(),
    records,
  ).dispatch_plan(1024)
  let dense_dispatch = projection_build_runtime_dispatch(
    dense_parameters,
    1024,
    projection_fixture_target(),
    records,
  ).dispatch_plan(1024)
  let qkv = row_variant_primary_test_source(qkv_parameters, qkv_dispatch)
  assert_true(qkv.contains("if (token_count == 1)"))
  assert_true(qkv.contains("__shfl_down_sync"))
  assert_true(qkv.contains("mma.sync.aligned.m16n8k16"))
  assert_true(qkv.contains("operand_index(local_row, k)"))
  assert_false(qkv.contains("wmma::"))
  let dense = row_variant_primary_test_source(dense_parameters, dense_dispatch)
  assert_true(dense.contains("if (token_count == 1)"))
  assert_true(dense.contains("__shfl_down_sync"))
  assert_true(dense.contains("mma.sync.aligned.m16n8k16"))
  assert_true(dense.contains("operand_index(local_row, k)"))
  assert_false(dense.contains("wmma::"))
  assert_eq(
    projection_dimensions(profile, qkv_parameters, qkv_dispatch).grid_x(),
    512,
  )
  assert_eq(
    projection_dimensions(profile, qkv_parameters, qkv_dispatch).shared_memory_bytes(),
    8192,
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

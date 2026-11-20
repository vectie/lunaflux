///|
fn require_qwen_fused_attention_plan_span(
  plan : @plan.ModelPlan,
  blueprint : PagedKvExecutionBlueprint,
  qkv_index : Int32,
) -> Unit raise FusedGroupedOperationError {
  let qk_norm_index = qkv_index + 1
  let rope_index = qkv_index + 2
  let attention_index = qkv_index + 3
  guard qkv_index >= 0 &&
    plan.operations().get(qkv_index) is Some(qkv) &&
    plan.operations().get(qk_norm_index) is Some(qk_norm) &&
    plan.operations().get(rope_index) is Some(rope) &&
    plan.operations().get(attention_index) is Some(attention) &&
    qkv.kind() == QkvProjection &&
    qk_norm.kind() == QkRmsNorm &&
    rope.kind() == RotaryEmbedding &&
    attention.kind() == CausalAttention &&
    qkv.outputs().get(0) is Some(qkv_output) &&
    qk_norm.inputs().get(0) is Some(Activation(qk_norm_input)) &&
    qk_norm.outputs().get(0) is Some(qk_norm_output) &&
    rope.inputs().get(0) is Some(Activation(rope_input)) &&
    rope.outputs().get(0) is Some(rope_output) &&
    attention.inputs().get(0) is Some(Activation(attention_input)) &&
    qkv_output == qk_norm_input &&
    qk_norm_output == rope_input &&
    rope_output == attention_input &&
    qkv.shape() is ProjectionShape(output_width~, ..) &&
    qk_norm.shape()
    is QkNormalizationShape(query_heads~, kv_heads~, head_dimension~, ..) &&
    rope.shape()
    is RotaryShape(
      query_heads=rope_query_heads,
      kv_heads=rope_kv_heads,
      head_dimension=rope_head_dimension,
      ..
    ) &&
    attention.shape()
    is AttentionShape(
      query_heads=attention_query_heads,
      kv_heads=attention_kv_heads,
      head_dimension=attention_head_dimension
    ) &&
    query_heads == rope_query_heads &&
    query_heads == attention_query_heads &&
    kv_heads == rope_kv_heads &&
    kv_heads == attention_kv_heads &&
    head_dimension == rope_head_dimension &&
    head_dimension == attention_head_dimension &&
    output_width == (query_heads + 2 * kv_heads) * head_dimension &&
    rope.kv_layer() == attention.kv_layer() &&
    blueprint.steps().get(qkv_index) is Some(qkv_step) &&
    blueprint.steps().get(qk_norm_index) is Some(qk_norm_step) &&
    blueprint.steps().get(rope_index) is Some(rope_step) &&
    blueprint.steps().get(attention_index) is Some(attention_step) &&
    qkv_step.operation_id() == qkv.id() &&
    qk_norm_step.operation_id() == qk_norm.id() &&
    rope_step.operation_id() == rope.id() &&
    attention_step.operation_id() == attention.id() &&
    qkv_step.operation_kind() == QkvProjection &&
    qk_norm_step.operation_kind() == QkRmsNorm &&
    rope_step.operation_kind() == RotaryEmbedding &&
    attention_step.operation_kind() == CausalAttention else {
    raise MismatchedFusedGroupedOperation(OperationOrder)
  }
}

///|
fn qwen_fused_attention_shapes_equal(
  operations : ReadOnlyArray[@plan.PlanOperation],
  seed_qkv : Int32,
  candidate_qkv : Int32,
) -> Bool {
  for offset in 0..<4 {
    guard operations.get(seed_qkv + offset) is Some(seed) &&
      operations.get(candidate_qkv + offset) is Some(candidate) else {
      return false
    }
    if seed.kind() != candidate.kind() || seed.shape() != candidate.shape() {
      return false
    }
  }
  true
}

///|
fn qwen_fused_binding_role_shape_equal(
  left : @launch_contract.AotOperandRole,
  right : @launch_contract.AotOperandRole,
) -> Bool {
  match (left, right) {
    (ExternalInput(a), ExternalInput(b)) => a == b
    (WeightInput(_), WeightInput(_)) => true
    (WeightScaleInput(_, _), WeightScaleInput(_, _)) => true
    (ActivationInput(_), ActivationInput(_)) => true
    (RuntimeInput(a), RuntimeInput(b)) => a == b
    (PersistentKvComponent(_, a), PersistentKvComponent(_, b)) => a == b
    (ActivationOutput(_), ActivationOutput(_)) => true
    (Workspace, Workspace) => true
    _ => false
  }
}

///|
fn qwen_fused_blueprint_step_shape_equal(
  seed : PagedKvExecutionStep,
  candidate : PagedKvExecutionStep,
) -> Bool {
  if seed.operation_kind() != candidate.operation_kind() ||
    seed.capability() != candidate.capability() ||
    seed.semantic_version() != candidate.semantic_version() ||
    seed.dimensions() != candidate.dimensions() ||
    seed.operands().length() != candidate.operands().length() {
    return false
  }
  for index, left in seed.operands() {
    let right = candidate.operands()[index]
    if !qwen_fused_binding_role_shape_equal(left.role(), right.role()) ||
      left.space() != right.space() ||
      left.byte_count() != right.byte_count() ||
      left.alignment() != right.alignment() {
      return false
    }
  }
  true
}

///|
fn require_qwen_fused_blueprint_span_shape(
  blueprint : PagedKvExecutionBlueprint,
  seed_qkv : Int32,
  candidate_qkv : Int32,
) -> Unit raise FusedGroupedOperationError {
  for offset in 0..<4 {
    guard blueprint.steps().get(seed_qkv + offset) is Some(seed) &&
      blueprint.steps().get(candidate_qkv + offset) is Some(candidate) &&
      qwen_fused_blueprint_step_shape_equal(seed, candidate) else {
      raise MismatchedFusedGroupedOperation(OperationOrder)
    }
  }
}

///|
fn qwen_partitioned_workspace_product(
  left : Int64,
  right : Int64,
) -> Int64 raise FusedGroupedOperationError {
  if left <= 0L || right <= 0L || left > 9223372036854775807L / right {
    raise MismatchedFusedGroupedOperation(Launch)
  }
  left * right
}

///|
fn qwen_partitioned_prefill_workspace_bytes(
  maximum_tokens : Int32,
  query_heads : Int32,
  head_dimension : Int32,
  partitions : Int32,
) -> Int64 raise FusedGroupedOperationError {
  let states = qwen_partitioned_workspace_product(
    qwen_partitioned_workspace_product(
      maximum_tokens.to_int64(),
      query_heads.to_int64(),
    ),
    partitions.to_int64(),
  )
  qwen_partitioned_workspace_product(
    qwen_partitioned_workspace_product(states, (head_dimension + 2).to_int64()),
    4L,
  )
}

///|
fn admit_qwen_partitioned_prefill_pair(
  modules : (
    @luna_fused_artifact_admission.ApprovedReusableFusedRuntimeModule,
    @luna_fused_artifact_admission.ApprovedReusableFusedRuntimeModule,
  )?,
  partial_abi : @luna_fused_artifact_admission.ReusableFusedRuntimeKernelAbi,
  merge_abi : @luna_fused_artifact_admission.ReusableFusedRuntimeKernelAbi,
  attention_operation : Int32,
  compiler_prefill_grid : Int32,
  maximum_tokens : Int32,
  query_heads : Int32,
  head_dimension : Int32,
) -> ApprovedQwenPartitionedPrefill? raise FusedGroupedOperationError {
  match modules {
    Some((partial, merge)) => {
      let partial_launch = partial.launch_dimensions()
      let merge_launch = merge.launch_dimensions()
      // AOT admission owns the checked worker geometry. The execution layer
      // binds semantic rows and matching partial/merge layouts, not a previous
      // backend's fixed 256-thread schedule.
      guard partial.query_tile_rows() is Some(query_tile_rows) &&
        merge.query_tile_rows() == Some(query_tile_rows) else {
        raise MismatchedFusedGroupedOperation(Launch)
      }
      if partial.kernel_abi() != partial_abi ||
        merge.kernel_abi() != merge_abi ||
        partial.operation_span() != (attention_operation, 1) ||
        merge.operation_span() != (attention_operation, 1) ||
        partial.module_digest() != merge.module_digest() ||
        partial.module_bytes() != merge.module_bytes() ||
        merge.query_metadata_version() != 0 ||
        query_tile_rows <= 0 ||
        partial_launch.grid_x() <= 0 ||
        partial_launch.grid_x() > compiler_prefill_grid ||
        partial_launch.grid_y() != query_heads ||
        partial_launch.grid_z() <= 1 ||
        partial_launch.block_y() != 1 ||
        partial_launch.block_z() != 1 ||
        partial_launch.shared_memory_bytes() <= 0 ||
        partial_launch.shared_memory_bytes() > 100000 ||
        merge_launch.grid_x() != partial_launch.grid_x() ||
        merge_launch.grid_y() != partial_launch.grid_y() ||
        merge_launch.grid_z() != 1 ||
        merge_launch.block_x() != partial_launch.block_x() ||
        merge_launch.block_y() != 1 ||
        merge_launch.block_z() != 1 ||
        merge_launch.shared_memory_bytes() != 0 {
        raise MismatchedFusedGroupedOperation(Launch)
      }
      Some({
        query_metadata: partial.query_metadata_version() == 1,
        module_digest: partial.module_digest(),
        module_bytes: partial.module_bytes(),
        partial_symbol: partial.function_symbol(),
        partial_launch,
        merge_symbol: merge.function_symbol(),
        merge_launch,
        query_tile_rows,
        workspace_bytes: qwen_partitioned_prefill_workspace_bytes(
          maximum_tokens,
          query_heads,
          head_dimension,
          partial_launch.grid_z(),
        ),
        workspace_alignment: 16L,
      })
    }
    None => None
  }
}

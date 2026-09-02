///|
fn require_qwen_fused_attention_plan_span(
  plan : @plan.ModelPlan,
  blueprint : PagedKvExecutionBlueprint,
  qkv_index : Int,
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
  seed_qkv : Int,
  candidate_qkv : Int,
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
  seed_qkv : Int,
  candidate_qkv : Int,
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
/// Expands one digest-pinned Qwen ingress V4/read-only V2 module pair over
/// every graph-identical QKV -> QkRmsNorm -> RoPE -> Attention layer span.
/// Parsing, model identity, shape matching, and blueprint joins all complete
/// during startup; the resulting spans require no token-step validation.
pub fn admit_reusable_qwen_fused_attention_spans(
  plan : @plan.ModelPlan,
  generation : @worker_protocol.ModelPlanGeneration,
  blueprint : PagedKvExecutionBlueprint,
  runtime : @luna_fused_artifact_admission.ApprovedReusableFusedRuntimeBundle,
) -> ReadOnlyArray[ApprovedQwenFusedAttentionSpan] raise FusedGroupedOperationError {
  if runtime.model_content_sha256() != plan.identity().content().as_hex() ||
    runtime.model_plan_sha256() != plan.identity().plan().as_hex() ||
    blueprint.identity() != plan.identity() ||
    runtime.target() != blueprint.target() {
    raise MismatchedFusedGroupedOperation(ModelIdentity)
  }
  guard runtime.qwen_qknorm_rope_kv_write_module() is Some(postprocess) &&
    runtime.qwen_paged_attention_prefill_module() is Some(readonly_module) &&
    runtime.qwen_paged_attention_decode_module() is Some(decode_module) &&
    postprocess.kernel_abi() == QwenQkNormPositionedRopePagedKvWriteProductionV2 &&
    (
      readonly_module.kernel_abi() == QwenPagedAttentionReadOnlyProductionV2 ||
      readonly_module.kernel_abi() ==
      QwenPagedAttentionReadOnlyDecodeSplitProductionV3
    ) &&
    (
      decode_module.kernel_abi() == QwenPagedAttentionReadOnlyProductionV2 ||
      decode_module.kernel_abi() ==
      QwenPagedAttentionReadOnlyDecodeSplitProductionV3
    ) &&
    runtime.qwen_qknorm_rope_kv_write_operation_indices()
    is Some((seed_qk_norm, seed_rope, seed_attention)) &&
    runtime.qwen_paged_attention_readonly_operation_index()
    is Some(readonly_attention) &&
    decode_module.operation_span() == (readonly_attention, 1) else {
    raise MismatchedFusedGroupedOperation(KernelAbi)
  }
  let seed_qkv = seed_qk_norm - 1
  if seed_qk_norm <= 0 ||
    seed_rope != seed_qk_norm + 1 ||
    seed_attention != seed_qk_norm + 2 ||
    readonly_attention != seed_attention {
    raise MismatchedFusedGroupedOperation(KernelAbi)
  }
  require_qwen_fused_attention_plan_span(plan, blueprint, seed_qkv)
  let phase_specific = readonly_module.module_digest() !=
    decode_module.module_digest() ||
    readonly_module.function_symbol() != decode_module.function_symbol() ||
    readonly_module.launch_dimensions() != decode_module.launch_dimensions()
  let readonly_launch = readonly_module.launch_dimensions()
  let decode_launch = decode_module.launch_dimensions()
  let maximum_rows = blueprint.profile().max_query_rows()
  let maximum_tokens = blueprint.profile().max_query_tokens()
  let compiler_prefill_grid = maximum_rows +
    (maximum_tokens - maximum_rows + 15) / 16
  guard plan.operations()[seed_qk_norm].shape()
    is QkNormalizationShape(query_heads~, kv_heads~, ..) &&
    kv_heads > 0 &&
    query_heads % kv_heads == 0 else {
    raise MismatchedFusedGroupedOperation(Launch)
  }
  let compiler_decode = decode_module.function_symbol() ==
    "lunaflux_attention_decode_tile_compiler_v1"
  let grouped_decode = compiler_decode &&
    decode_launch.grid_y() == kv_heads &&
    decode_launch.block_x() == query_heads / kv_heads * 32 &&
    decode_launch.shared_memory_bytes() > 0
  let direct_decode = compiler_decode &&
    decode_launch.grid_y() == query_heads &&
    decode_launch.block_x() == 32 &&
    decode_launch.shared_memory_bytes() == 0
  let valid_prefill_launch = if phase_specific {
    readonly_module.kernel_abi() == QwenPagedAttentionReadOnlyProductionV2 &&
    decode_module.kernel_abi() ==
    QwenPagedAttentionReadOnlyDecodeSplitProductionV3 &&
    readonly_launch.grid_x() == compiler_prefill_grid &&
    readonly_launch.block_x() == 256 &&
    readonly_launch.shared_memory_bytes() > 0 &&
    readonly_launch.shared_memory_bytes() <= 100000
  } else {
    readonly_launch.grid_x() == Int::min(maximum_tokens, 128) &&
    readonly_launch.block_x() == 128 &&
    readonly_launch.shared_memory_bytes() == 0
  }
  let valid_decode_launch = if grouped_decode {
    decode_launch.grid_x() == maximum_rows &&
    decode_launch.grid_y() == kv_heads &&
    decode_launch.block_x() == query_heads / kv_heads * 32 &&
    decode_launch.shared_memory_bytes() > 0 &&
    decode_launch.shared_memory_bytes() <= 100000
  } else if direct_decode {
    decode_launch.grid_x() == maximum_rows &&
    decode_launch.grid_y() == query_heads &&
    decode_launch.block_x() == 32 &&
    decode_launch.shared_memory_bytes() == 0
  } else {
    decode_launch.grid_x() == Int::min(maximum_tokens, 128) &&
    decode_launch.block_x() == 128 &&
    decode_launch.shared_memory_bytes() == 0
  }
  guard maximum_tokens >= maximum_rows &&
    valid_prefill_launch &&
    readonly_launch.grid_y() == query_heads &&
    readonly_launch.grid_z() == 1 &&
    readonly_launch.block_y() == 1 &&
    readonly_launch.block_z() == 1 &&
    (grouped_decode || direct_decode || decode_launch.grid_y() == query_heads) &&
    decode_launch.grid_z() == 1 &&
    valid_decode_launch &&
    decode_launch.block_y() == 1 &&
    decode_launch.block_z() == 1 &&
    postprocess.launch_dimensions().grid_x() ==
    Int::min(blueprint.profile().max_query_tokens(), 4096) &&
    postprocess.launch_dimensions().grid_y() == query_heads + 2 * kv_heads &&
    postprocess.launch_dimensions().grid_z() == 1 &&
    postprocess.launch_dimensions().block_x() == 128 &&
    postprocess.launch_dimensions().block_y() == 1 &&
    postprocess.launch_dimensions().block_z() == 1 &&
    postprocess.launch_dimensions().shared_memory_bytes() == 0 else {
    raise MismatchedFusedGroupedOperation(Launch)
  }
  let operations = plan.operations()
  let spans : Array[ApprovedQwenFusedAttentionSpan] = []
  if operations.length() < 4 {
    raise MismatchedFusedGroupedOperation(OperationOrder)
  }
  for qkv_index in 0..<(operations.length() - 3) {
    if !qwen_fused_attention_shapes_equal(operations, seed_qkv, qkv_index) {
      continue
    }
    require_qwen_fused_attention_plan_span(plan, blueprint, qkv_index)
    require_qwen_fused_blueprint_span_shape(blueprint, seed_qkv, qkv_index)
    spans.push({
      generation,
      target: runtime.target(),
      runtime_digest: runtime.source_digest(),
      postprocess_module_digest: postprocess.module_digest(),
      postprocess_module_bytes: postprocess.module_bytes(),
      postprocess_symbol: postprocess.function_symbol(),
      postprocess_launch: postprocess.launch_dimensions(),
      readonly_module_digest: readonly_module.module_digest(),
      readonly_module_bytes: readonly_module.module_bytes(),
      readonly_symbol: readonly_module.function_symbol(),
      readonly_launch: readonly_module.launch_dimensions(),
      decode_module_digest: decode_module.module_digest(),
      decode_module_bytes: decode_module.module_bytes(),
      decode_symbol: decode_module.function_symbol(),
      decode_launch: decode_module.launch_dimensions(),
      qkv_index,
      qk_norm_index: qkv_index + 1,
      rope_index: qkv_index + 2,
      attention_index: qkv_index + 3,
    })
  }
  if spans.is_empty() {
    raise MismatchedFusedGroupedOperation(OperationOrder)
  }
  ReadOnlyArray::from_array(spans)
}

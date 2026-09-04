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
  maximum_tokens : Int,
  query_heads : Int,
  head_dimension : Int,
  partitions : Int,
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
  attention_operation : Int,
  compiler_prefill_grid : Int,
  maximum_tokens : Int,
  query_heads : Int,
  head_dimension : Int,
  query_tile_rows : Int,
) -> ApprovedQwenPartitionedPrefill? raise FusedGroupedOperationError {
  match modules {
    Some((partial, merge)) => {
      let partial_launch = partial.launch_dimensions()
      let merge_launch = merge.launch_dimensions()
      if partial.kernel_abi() != partial_abi ||
        merge.kernel_abi() != merge_abi ||
        partial.operation_span() != (attention_operation, 1) ||
        merge.operation_span() != (attention_operation, 1) ||
        partial.module_digest() != merge.module_digest() ||
        partial.module_bytes() != merge.module_bytes() ||
        query_tile_rows <= 0 ||
        partial_launch.grid_x() <= 0 ||
        partial_launch.grid_x() > compiler_prefill_grid ||
        partial_launch.grid_y() != query_heads ||
        partial_launch.grid_z() <= 1 ||
        partial_launch.block_x() != 256 ||
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
  let ingress_selection = match runtime.qwen_full_attention_ingress_module() {
    Some(module_) => Some((module_, true))
    None =>
      match runtime.qwen_qknorm_rope_kv_write_module() {
        Some(module_) => Some((module_, false))
        None => None
      }
  }
  let partitioned_modules = match
    (
      runtime.qwen_partitioned_prefill_partial_module(),
      runtime.qwen_partitioned_prefill_merge_module(),
    ) {
    (Some(partial), Some(merge)) => Some((partial, merge))
    (None, None) => None
    _ => raise MismatchedFusedGroupedOperation(KernelAbi)
  }
  let deep_partitioned_modules = match
    (
      runtime.qwen_partitioned8_prefill_partial_module(),
      runtime.qwen_partitioned8_prefill_merge_module(),
    ) {
    (Some(partial), Some(merge)) => Some((partial, merge))
    (None, None) => None
    _ => raise MismatchedFusedGroupedOperation(KernelAbi)
  }
  let prefill_module = runtime.qwen_paged_attention_prefill_module()
  guard ingress_selection is Some((postprocess, includes_projection)) &&
    prefill_module is Some(readonly_module) &&
    runtime.qwen_paged_attention_decode_module() is Some(decode_module) &&
    (
      (
        includes_projection &&
        postprocess.kernel_abi() ==
        QwenQkvQkNormPositionedRopePagedKvWriteProductionV4 &&
        postprocess.operation_count() == 4
      ) ||
      (
        !includes_projection &&
        postprocess.kernel_abi() ==
        QwenQkNormPositionedRopePagedKvWriteProductionV2 &&
        postprocess.operation_count() == 3
      )
    ) &&
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
    runtime.qwen_paged_attention_readonly_operation_index()
    is Some(readonly_attention) &&
    decode_module.operation_span() == (readonly_attention, 1) else {
    raise MismatchedFusedGroupedOperation(KernelAbi)
  }
  let seed_qkv = if includes_projection {
    postprocess.first_operation_index()
  } else {
    postprocess.first_operation_index() - 1
  }
  let seed_qk_norm = seed_qkv + 1
  let seed_rope = seed_qkv + 2
  let seed_attention = seed_qkv + 3
  if seed_qkv < 0 ||
    postprocess.first_operation_index() !=
    (if includes_projection { seed_qkv } else { seed_qk_norm }) ||
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
    is QkNormalizationShape(query_heads~, kv_heads~, head_dimension~, ..) &&
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
  let split_grouped_decode = compiler_decode &&
    decode_launch.grid_y() == kv_heads &&
    decode_launch.block_x() > query_heads / kv_heads * 32 &&
    decode_launch.block_x() <= 256 &&
    decode_launch.block_x() % (query_heads / kv_heads * 32) == 0 &&
    decode_launch.shared_memory_bytes() > 0
  let wide_query_prefill = match runtime.qwen_wide_query_prefill_module() {
    Some(module_) => {
      let launch = module_.launch_dimensions()
      if module_.module_digest() == readonly_module.module_digest() &&
        module_.function_symbol() == readonly_module.function_symbol() &&
        launch == readonly_launch {
        // A stable bundle slot may carry the selected prefill module when the
        // semantic profile has no strictly wider query tile. Do not build a
        // duplicate executor or route a bucket through a fake specialization.
        None
      } else if module_.kernel_abi() !=
        QwenPagedAttentionPrefillWideQueryProductionV5 ||
        module_.operation_span() != (seed_attention, 1) ||
        launch.grid_x() != compiler_prefill_grid ||
        launch.grid_y() != query_heads ||
        launch.grid_z() != 1 ||
        launch.block_x() != 256 ||
        launch.block_y() != 1 ||
        launch.block_z() != 1 ||
        launch.shared_memory_bytes() <= 0 ||
        launch.shared_memory_bytes() > 100000 {
        raise MismatchedFusedGroupedOperation(Launch)
      } else {
        Some({
          module_digest: module_.module_digest(),
          module_bytes: module_.module_bytes(),
          symbol: module_.function_symbol(),
          launch,
          query_tile_rows: 64,
        })
      }
    }
    None => None
  }
  let partitioned_prefill = admit_qwen_partitioned_prefill_pair(
    partitioned_modules,
    QwenPagedAttentionPrefillPartitionedPartialProductionV4,
    QwenPagedAttentionPrefillPartitionedMergeProductionV4,
    seed_attention,
    compiler_prefill_grid,
    maximum_tokens,
    query_heads,
    head_dimension,
    16,
  )
  let deep_partitioned_prefill = admit_qwen_partitioned_prefill_pair(
    deep_partitioned_modules,
    QwenPagedAttentionPrefillPartitioned8PartialProductionV6,
    QwenPagedAttentionPrefillPartitioned8MergeProductionV6,
    seed_attention,
    compiler_prefill_grid,
    maximum_tokens,
    query_heads,
    head_dimension,
    16,
  )
  match (partitioned_prefill, deep_partitioned_prefill) {
    (Some(base), Some(deep)) =>
      if base.partial_launch.grid_z() >= deep.partial_launch.grid_z() {
        raise MismatchedFusedGroupedOperation(Launch)
      }
    (None, Some(_)) => raise MismatchedFusedGroupedOperation(KernelAbi)
    _ => ()
  }
  let valid_prefill_launch = if partitioned_prefill is Some(_) {
    true
  } else if phase_specific {
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
  } else if split_grouped_decode {
    decode_launch.grid_x() == maximum_rows &&
    decode_launch.grid_y() == kv_heads &&
    decode_launch.block_x() <= 256 &&
    decode_launch.shared_memory_bytes() > 0 &&
    decode_launch.shared_memory_bytes() <= 100000
  } else {
    decode_launch.grid_x() == Int::min(maximum_tokens, 128) &&
    decode_launch.block_x() == 128 &&
    decode_launch.shared_memory_bytes() == 0
  }
  let ingress_grid_x = if includes_projection {
    (blueprint.profile().max_query_tokens() + 15) / 16
  } else {
    Int::min(blueprint.profile().max_query_tokens(), 4096)
  }
  guard maximum_tokens >= maximum_rows &&
    valid_prefill_launch &&
    readonly_launch.grid_y() == query_heads &&
    readonly_launch.grid_z() == 1 &&
    readonly_launch.block_y() == 1 &&
    readonly_launch.block_z() == 1 &&
    (
      grouped_decode ||
      split_grouped_decode ||
      direct_decode ||
      decode_launch.grid_y() == query_heads
    ) &&
    decode_launch.grid_z() == 1 &&
    valid_decode_launch &&
    decode_launch.block_y() == 1 &&
    decode_launch.block_z() == 1 &&
    postprocess.launch_dimensions().grid_x() == ingress_grid_x &&
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
      ingress_includes_projection: includes_projection,
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
      wide_query_prefill,
      partitioned_prefill,
      deep_partitioned_prefill,
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

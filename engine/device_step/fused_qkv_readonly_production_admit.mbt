///|
/// Joins the exact production V2 artifact pair to the model blueprint and the
/// live opened device. This is startup admission only; the returned span does
/// no hashing, device probing, or policy work during token execution.
pub fn admit_production_fused_qkv_readonly_attention_span(
  plan : @plan.ModelPlan,
  generation : @worker_protocol.ModelPlanGeneration,
  blueprint : PagedKvExecutionBlueprint,
  artifacts : @luna_paged_composite_artifact_admission.ApprovedFusedQkvReadOnlyProductionArtifacts,
  live_device : @device.ContextDeviceIdentity,
) -> ApprovedFusedQkvReadOnlyAttentionSpan raise FusedGroupedOperationError {
  let catalog = artifacts.fused_catalog()
  let fused_qkv = artifacts.fused_bundle().qkv_module()
  let readonly_identity = artifacts.readonly_identity()
  let kv_write = artifacts.kv_write_fallback()
  require_exact_fused_catalog_entry(catalog, fused_qkv)
  if fused_qkv.family() != QkvPositionedRopePagedKvWrite ||
    fused_qkv.artifact_purpose() != ProductionFastPath ||
    fused_qkv.kernel_abi() != QkvPositionedRopePagedKvWriteProductionAbiV2 ||
    fused_qkv.dispatch_canary_per_token() != 0U ||
    readonly_identity.kernel_abi() !=
    RotatedQueryPagedKvReadOnlyProductionRawPointerV2 ||
    readonly_identity.dispatch_canary_per_token() != 0 ||
    readonly_identity.dispatch_canary_cell_count() != 0 {
    raise InvalidFusedGroupedOperation(KernelAbi)
  }
  if blueprint.identity() != plan.identity() ||
    blueprint.target() != catalog.target() ||
    readonly_identity.target() != blueprint.target() ||
    kv_write.target() != blueprint.target() {
    raise MismatchedFusedGroupedOperation(Target)
  }
  let fallbacks = fused_qkv.fallbacks()
  if fallbacks.length() != 3 {
    raise MismatchedFusedGroupedOperation(StandaloneFallback)
  }
  let qkv_index = fallbacks[0].operation_index()
  let rope_index = fallbacks[1].operation_index()
  let attention_index = fallbacks[2].operation_index()
  let (positioned_qkv, full_qkv_row_width) = require_fused_qkv_readonly_plan_span(
    plan, qkv_index, rope_index, attention_index,
  )
  guard blueprint.steps().get(qkv_index) is Some(qkv_step) &&
    blueprint.steps().get(rope_index) is Some(rope_step) &&
    blueprint.steps().get(attention_index) is Some(attention_step) &&
    qkv_step.operation_kind() == QkvProjection &&
    rope_step.operation_kind() == RotaryEmbedding &&
    attention_step.operation_kind() == CausalAttention else {
    raise MismatchedFusedGroupedOperation(OperationOrder)
  }
  if readonly_identity.operation_id() != attention_index ||
    readonly_identity.activation_ref() != positioned_qkv.as_int() ||
    readonly_identity.input_row_width() != full_qkv_row_width ||
    kv_write.operation_id().as_int() != attention_index ||
    kv_write.positioned_qkv_activation() != positioned_qkv ||
    kv_write.positioned_qkv_row_width() != full_qkv_row_width {
    raise MismatchedFusedGroupedOperation(Activation)
  }
  let profile = blueprint.profile()
  let readonly_shape = readonly_identity.profile_shape()
  if readonly_identity.profile_id() != profile.id().as_int() ||
    readonly_shape !=
    (
      profile.max_query_rows(),
      profile.max_query_tokens(),
      profile.max_page_table_entries(),
    ) ||
    kv_write.profile() != profile {
    raise MismatchedFusedGroupedOperation(Profile)
  }
  let layout = blueprint.layout()
  if readonly_identity.tokens_per_page() != layout.tokens_per_page() ||
    readonly_identity.page_stride_bytes() != layout.page_stride_bytes() ||
    readonly_identity.physical_page_capacity() !=
    layout.total_page_count().to_int64() ||
    kv_write.tokens_per_page() != layout.tokens_per_page() ||
    kv_write.total_page_count() != layout.total_page_count() ||
    kv_write.page_stride_bytes() != layout.page_stride_bytes() {
    raise MismatchedFusedGroupedOperation(KvLayout)
  }
  let observed = live_device.capability()
  let expected = readonly_identity.target()
  if readonly_identity.approved_device_uuid() != live_device.uuid() ||
    readonly_identity.approved_device_pci_bus_id() != live_device.pci() ||
    readonly_identity.approved_device_name_hex() !=
    @crypto.bytes_to_hex_string(@utf8.encode(live_device.name())) ||
    readonly_identity.approved_device_total_memory_bytes() !=
    live_device.total_memory_bytes() ||
    observed.compute_major() != expected.compute_major() ||
    observed.compute_minor() != expected.compute_minor() ||
    observed.supports_bf16() != expected.supports_bf16() ||
    observed.supports_cublas_lt() != expected.supports_cublas_lt() {
    raise MismatchedFusedGroupedOperation(RuntimeDevice)
  }
  {
    generation,
    target: blueprint.target(),
    qkv_module_bytes: fused_qkv.module_bytes(),
    qkv_symbol: fused_qkv.function_symbol(),
    qkv_launch: fused_qkv.launch_dimensions(),
    readonly_module_bytes: readonly_identity.module_bytes(),
    readonly_symbol: readonly_identity.function_symbol(),
    readonly_launch: readonly_identity.launch_dimensions(),
    qkv_index,
    rope_index,
    attention_index,
  }
}

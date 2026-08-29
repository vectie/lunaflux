///|
pub fn ApprovedPagedAttentionReadOnlyRuntimeIdentity::manifest_digest_sha256(
  self : ApprovedPagedAttentionReadOnlyRuntimeIdentity,
) -> String {
  self.manifest
}

///|
pub fn ApprovedPagedAttentionReadOnlyRuntimeIdentity::catalog_digest_sha256(
  self : ApprovedPagedAttentionReadOnlyRuntimeIdentity,
) -> String {
  self.catalog
}

///|
pub fn ApprovedPagedAttentionReadOnlyRuntimeIdentity::candidate_digest_sha256(
  self : ApprovedPagedAttentionReadOnlyRuntimeIdentity,
) -> String {
  self.candidate
}

///|
pub fn ApprovedPagedAttentionReadOnlyRuntimeIdentity::source_digest_sha256(
  self : ApprovedPagedAttentionReadOnlyRuntimeIdentity,
) -> String {
  self.source
}

///|
pub fn ApprovedPagedAttentionReadOnlyRuntimeIdentity::recipe_digest_sha256(
  self : ApprovedPagedAttentionReadOnlyRuntimeIdentity,
) -> String {
  self.recipe
}

///|
pub fn ApprovedPagedAttentionReadOnlyRuntimeIdentity::fallback_source_digest_sha256(
  self : ApprovedPagedAttentionReadOnlyRuntimeIdentity,
) -> String {
  self.fallback_source
}

///|
pub fn ApprovedPagedAttentionReadOnlyRuntimeIdentity::fallback_recipe_digest_sha256(
  self : ApprovedPagedAttentionReadOnlyRuntimeIdentity,
) -> String {
  self.fallback_recipe
}

///|
pub fn ApprovedPagedAttentionReadOnlyRuntimeIdentity::compiled_binding_digest_sha256(
  self : ApprovedPagedAttentionReadOnlyRuntimeIdentity,
) -> String {
  self.compiled_binding
}

///|
pub fn ApprovedPagedAttentionReadOnlyRuntimeIdentity::module_digest_sha256(
  self : ApprovedPagedAttentionReadOnlyRuntimeIdentity,
) -> String {
  self.module_digest
}

///|
pub fn ApprovedPagedAttentionReadOnlyRuntimeIdentity::physical_evidence_digest_sha256(
  self : ApprovedPagedAttentionReadOnlyRuntimeIdentity,
) -> String {
  self.physical_evidence
}

///|
pub fn ApprovedPagedAttentionReadOnlyRuntimeIdentity::approved_policy_digest_sha256(
  self : ApprovedPagedAttentionReadOnlyRuntimeIdentity,
) -> String {
  self.approved_policy
}

///|
pub fn ApprovedPagedAttentionReadOnlyRuntimeIdentity::toolchain_digest_sha256(
  self : ApprovedPagedAttentionReadOnlyRuntimeIdentity,
) -> String {
  self.toolchain
}

///|
pub fn ApprovedPagedAttentionReadOnlyRuntimeIdentity::driver_identity_sha256(
  self : ApprovedPagedAttentionReadOnlyRuntimeIdentity,
) -> String {
  self.driver
}

///|
pub fn ApprovedPagedAttentionReadOnlyRuntimeIdentity::target(
  self : ApprovedPagedAttentionReadOnlyRuntimeIdentity,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn ApprovedPagedAttentionReadOnlyRuntimeIdentity::operation_id(
  self : ApprovedPagedAttentionReadOnlyRuntimeIdentity,
) -> Int {
  self.operation
}

///|
pub fn ApprovedPagedAttentionReadOnlyRuntimeIdentity::activation_ref(
  self : ApprovedPagedAttentionReadOnlyRuntimeIdentity,
) -> Int {
  self.activation
}

///|
pub fn ApprovedPagedAttentionReadOnlyRuntimeIdentity::input_row_width(
  self : ApprovedPagedAttentionReadOnlyRuntimeIdentity,
) -> Int {
  self.input_row_width
}

///|
pub fn ApprovedPagedAttentionReadOnlyRuntimeIdentity::profile_id(
  self : ApprovedPagedAttentionReadOnlyRuntimeIdentity,
) -> Int {
  self.profile_id
}

///|
pub fn ApprovedPagedAttentionReadOnlyRuntimeIdentity::profile_shape(
  self : ApprovedPagedAttentionReadOnlyRuntimeIdentity,
) -> (Int, Int, Int) {
  (self.profile_rows, self.profile_tokens, self.profile_page_entries)
}

///|
pub fn ApprovedPagedAttentionReadOnlyRuntimeIdentity::tokens_per_page(
  self : ApprovedPagedAttentionReadOnlyRuntimeIdentity,
) -> Int {
  self.tokens_per_page
}

///|
pub fn ApprovedPagedAttentionReadOnlyRuntimeIdentity::page_stride_bytes(
  self : ApprovedPagedAttentionReadOnlyRuntimeIdentity,
) -> Int64 {
  self.page_stride_bytes
}

///|
pub fn ApprovedPagedAttentionReadOnlyRuntimeIdentity::physical_page_capacity(
  self : ApprovedPagedAttentionReadOnlyRuntimeIdentity,
) -> Int64 {
  self.physical_page_capacity
}

///|
pub fn ApprovedPagedAttentionReadOnlyRuntimeIdentity::function_symbol(
  self : ApprovedPagedAttentionReadOnlyRuntimeIdentity,
) -> String {
  self.symbol
}

///|
pub fn ApprovedPagedAttentionReadOnlyRuntimeIdentity::kernel_abi(
  self : ApprovedPagedAttentionReadOnlyRuntimeIdentity,
) -> @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyKernelAbi {
  self.abi
}

///|
pub fn ApprovedPagedAttentionReadOnlyRuntimeIdentity::launch_dimensions(
  self : ApprovedPagedAttentionReadOnlyRuntimeIdentity,
) -> @launch_contract.AotLaunchDimensions {
  self.launch
}

///|
pub fn ApprovedPagedAttentionReadOnlyRuntimeIdentity::module_bytes(
  self : ApprovedPagedAttentionReadOnlyRuntimeIdentity,
) -> Bytes {
  self.module_bytes
}

///|
pub fn ApprovedPagedAttentionReadOnlyRuntimeIdentity::dispatch_canary_per_token(
  self : ApprovedPagedAttentionReadOnlyRuntimeIdentity,
) -> Int {
  self.dispatch_canary
}

///|
pub fn ApprovedPagedAttentionReadOnlyRuntimeIdentity::dispatch_canary_cell_count(
  self : ApprovedPagedAttentionReadOnlyRuntimeIdentity,
) -> Int {
  self.dispatch_canary_cells
}

///|
pub fn ApprovedPagedAttentionReadOnlyRuntimeIdentity::approved_device_uuid(
  self : ApprovedPagedAttentionReadOnlyRuntimeIdentity,
) -> String {
  self.device_uuid
}

///|
pub fn ApprovedPagedAttentionReadOnlyRuntimeIdentity::approved_device_pci_bus_id(
  self : ApprovedPagedAttentionReadOnlyRuntimeIdentity,
) -> String {
  self.device_pci
}

///|
pub fn ApprovedPagedAttentionReadOnlyRuntimeIdentity::approved_device_name_hex(
  self : ApprovedPagedAttentionReadOnlyRuntimeIdentity,
) -> String {
  self.device_name_hex
}

///|
pub fn ApprovedPagedAttentionReadOnlyRuntimeIdentity::approved_device_total_memory_bytes(
  self : ApprovedPagedAttentionReadOnlyRuntimeIdentity,
) -> Int64 {
  self.device_memory_bytes
}

///|
pub fn ApprovedPagedAttentionReadOnlyRuntimeIdentity::runtime_device_digest_sha256(
  self : ApprovedPagedAttentionReadOnlyRuntimeIdentity,
) -> String {
  self.runtime_device
}

///|
pub fn QualifiedPagedKvWriteFallbackIdentity::model_content_digest_sha256(
  self : QualifiedPagedKvWriteFallbackIdentity,
) -> String {
  self.model_content
}

///|
pub fn QualifiedPagedKvWriteFallbackIdentity::model_plan_digest_sha256(
  self : QualifiedPagedKvWriteFallbackIdentity,
) -> String {
  self.model_plan
}

///|
pub fn QualifiedPagedKvWriteFallbackIdentity::candidate_digest_sha256(
  self : QualifiedPagedKvWriteFallbackIdentity,
) -> String {
  self.candidate
}

///|
pub fn QualifiedPagedKvWriteFallbackIdentity::source_digest_sha256(
  self : QualifiedPagedKvWriteFallbackIdentity,
) -> String {
  self.source
}

///|
pub fn QualifiedPagedKvWriteFallbackIdentity::recipe_digest_sha256(
  self : QualifiedPagedKvWriteFallbackIdentity,
) -> String {
  self.recipe
}

///|
pub fn QualifiedPagedKvWriteFallbackIdentity::compiled_binding_digest_sha256(
  self : QualifiedPagedKvWriteFallbackIdentity,
) -> String {
  self.compiled_binding
}

///|
pub fn QualifiedPagedKvWriteFallbackIdentity::module_digest_sha256(
  self : QualifiedPagedKvWriteFallbackIdentity,
) -> String {
  self.module_digest
}

///|
pub fn QualifiedPagedKvWriteFallbackIdentity::physical_evidence_digest_sha256(
  self : QualifiedPagedKvWriteFallbackIdentity,
) -> String {
  self.physical_evidence
}

///|
pub fn QualifiedPagedKvWriteFallbackIdentity::approved_policy_digest_sha256(
  self : QualifiedPagedKvWriteFallbackIdentity,
) -> String {
  self.approved_policy
}

///|
pub fn QualifiedPagedKvWriteFallbackIdentity::fallback_source_digest_sha256(
  self : QualifiedPagedKvWriteFallbackIdentity,
) -> String {
  self.fallback_source
}

///|
pub fn QualifiedPagedKvWriteFallbackIdentity::fallback_recipe_digest_sha256(
  self : QualifiedPagedKvWriteFallbackIdentity,
) -> String {
  self.fallback_recipe
}

///|
pub fn QualifiedPagedKvWriteFallbackIdentity::operation_id(
  self : QualifiedPagedKvWriteFallbackIdentity,
) -> @plan.OperationId {
  self.operation
}

///|
pub fn QualifiedPagedKvWriteFallbackIdentity::positioned_qkv_activation(
  self : QualifiedPagedKvWriteFallbackIdentity,
) -> @plan.ActivationRef {
  self.activation
}

///|
pub fn QualifiedPagedKvWriteFallbackIdentity::positioned_qkv_row_width(
  self : QualifiedPagedKvWriteFallbackIdentity,
) -> Int {
  self.row_width
}

///|
pub fn QualifiedPagedKvWriteFallbackIdentity::target(
  self : QualifiedPagedKvWriteFallbackIdentity,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn QualifiedPagedKvWriteFallbackIdentity::profile(
  self : QualifiedPagedKvWriteFallbackIdentity,
) -> @launch_contract.PagedKernelProfile {
  self.profile
}

///|
pub fn QualifiedPagedKvWriteFallbackIdentity::tokens_per_page(
  self : QualifiedPagedKvWriteFallbackIdentity,
) -> Int {
  self.tokens_per_page
}

///|
pub fn QualifiedPagedKvWriteFallbackIdentity::total_page_count(
  self : QualifiedPagedKvWriteFallbackIdentity,
) -> Int {
  self.total_page_count
}

///|
pub fn QualifiedPagedKvWriteFallbackIdentity::page_stride_bytes(
  self : QualifiedPagedKvWriteFallbackIdentity,
) -> Int64 {
  self.page_stride_bytes
}

///|
pub fn QualifiedPagedKvWriteFallbackIdentity::toolchain_digest_sha256(
  self : QualifiedPagedKvWriteFallbackIdentity,
) -> String {
  self.toolchain
}

///|
pub fn QualifiedPagedKvWriteFallbackIdentity::driver_identity_sha256(
  self : QualifiedPagedKvWriteFallbackIdentity,
) -> String {
  self.driver
}

///|
pub fn ApprovedFusedQkvReadOnlyProductionArtifacts::fused_catalog(
  self : ApprovedFusedQkvReadOnlyProductionArtifacts,
) -> @luna_fused_artifact_admission.ApprovedFusedKernelCatalog {
  self.fused_catalog
}

///|
pub fn ApprovedFusedQkvReadOnlyProductionArtifacts::fused_bundle(
  self : ApprovedFusedQkvReadOnlyProductionArtifacts,
) -> @luna_fused_artifact_admission.ApprovedFusedKernelBundle {
  self.fused_bundle
}

///|
pub fn ApprovedFusedQkvReadOnlyProductionArtifacts::readonly_identity(
  self : ApprovedFusedQkvReadOnlyProductionArtifacts,
) -> ApprovedPagedAttentionReadOnlyRuntimeIdentity {
  self.readonly_identity
}

///|
pub fn ApprovedFusedQkvReadOnlyProductionArtifacts::kv_write_fallback(
  self : ApprovedFusedQkvReadOnlyProductionArtifacts,
) -> QualifiedPagedKvWriteFallbackIdentity {
  self.kv_write
}

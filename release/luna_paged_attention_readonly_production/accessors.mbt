///|
pub fn ApprovedPagedAttentionReadOnlyCatalog::manifest_digest(
  self : ApprovedPagedAttentionReadOnlyCatalog,
) -> ReadOnlyProductionDigest {
  self.manifest.digest
}

///|
pub fn ApprovedPagedAttentionReadOnlyCatalog::digest(
  self : ApprovedPagedAttentionReadOnlyCatalog,
) -> ReadOnlyProductionDigest {
  self.digest
}

///|
pub fn ApprovedPagedAttentionReadOnlyCatalog::target(
  self : ApprovedPagedAttentionReadOnlyCatalog,
) -> @catalog.DeviceTarget {
  self.module_.identity.target
}

///|
pub fn ApprovedPagedAttentionReadOnlyModule::manifest_digest(
  self : ApprovedPagedAttentionReadOnlyModule,
) -> ReadOnlyProductionDigest {
  self.manifest
}

///|
pub fn ApprovedPagedAttentionReadOnlyModule::candidate_digest_sha256(
  self : ApprovedPagedAttentionReadOnlyModule,
) -> String {
  self.identity.candidate
}

///|
pub fn ApprovedPagedAttentionReadOnlyModule::source_digest_sha256(
  self : ApprovedPagedAttentionReadOnlyModule,
) -> String {
  self.identity.source
}

///|
pub fn ApprovedPagedAttentionReadOnlyModule::recipe_digest_sha256(
  self : ApprovedPagedAttentionReadOnlyModule,
) -> String {
  self.identity.recipe
}

///|
pub fn ApprovedPagedAttentionReadOnlyModule::fallback_source_digest_sha256(
  self : ApprovedPagedAttentionReadOnlyModule,
) -> String {
  self.identity.fallback_source
}

///|
pub fn ApprovedPagedAttentionReadOnlyModule::fallback_recipe_digest_sha256(
  self : ApprovedPagedAttentionReadOnlyModule,
) -> String {
  self.identity.fallback_recipe
}

///|
pub fn ApprovedPagedAttentionReadOnlyModule::compile_receipt_digest_sha256(
  self : ApprovedPagedAttentionReadOnlyModule,
) -> String {
  self.identity.compile_receipt
}

///|
pub fn ApprovedPagedAttentionReadOnlyModule::physical_files_manifest_digest_sha256(
  self : ApprovedPagedAttentionReadOnlyModule,
) -> String {
  self.identity.physical_files_manifest
}

///|
pub fn ApprovedPagedAttentionReadOnlyModule::physical_outer_seal_digest_sha256(
  self : ApprovedPagedAttentionReadOnlyModule,
) -> String {
  self.identity.physical_outer_seal
}

///|
pub fn ApprovedPagedAttentionReadOnlyModule::compiled_binding_digest_sha256(
  self : ApprovedPagedAttentionReadOnlyModule,
) -> String {
  self.identity.compiled_binding
}

///|
pub fn ApprovedPagedAttentionReadOnlyModule::module_digest_sha256(
  self : ApprovedPagedAttentionReadOnlyModule,
) -> String {
  self.identity.module_digest
}

///|
pub fn ApprovedPagedAttentionReadOnlyModule::physical_evidence_digest_sha256(
  self : ApprovedPagedAttentionReadOnlyModule,
) -> String {
  self.identity.physical_evidence
}

///|
pub fn ApprovedPagedAttentionReadOnlyModule::approved_physical_policy_digest_sha256(
  self : ApprovedPagedAttentionReadOnlyModule,
) -> String {
  self.identity.approved_physical_policy
}

///|
pub fn ApprovedPagedAttentionReadOnlyModule::toolchain_digest_sha256(
  self : ApprovedPagedAttentionReadOnlyModule,
) -> String {
  self.identity.toolchain
}

///|
pub fn ApprovedPagedAttentionReadOnlyModule::driver_identity_sha256(
  self : ApprovedPagedAttentionReadOnlyModule,
) -> String {
  self.identity.driver
}

///|
pub fn ApprovedPagedAttentionReadOnlyModule::approved_device_uuid(
  self : ApprovedPagedAttentionReadOnlyModule,
) -> String {
  self.identity.device_uuid
}

///|
pub fn ApprovedPagedAttentionReadOnlyModule::approved_device_pci_bus_id(
  self : ApprovedPagedAttentionReadOnlyModule,
) -> String {
  self.identity.device_pci_bus_id
}

///|
pub fn ApprovedPagedAttentionReadOnlyModule::approved_device_name_hex(
  self : ApprovedPagedAttentionReadOnlyModule,
) -> String {
  self.identity.device_name_hex
}

///|
pub fn ApprovedPagedAttentionReadOnlyModule::approved_device_total_memory_bytes(
  self : ApprovedPagedAttentionReadOnlyModule,
) -> Int64 {
  try! readonly_exact_int64(
    self.identity.device_total_memory_bytes,
    PhysicalPolicy,
  )
}

///|
pub fn ApprovedPagedAttentionReadOnlyModule::target(
  self : ApprovedPagedAttentionReadOnlyModule,
) -> @catalog.DeviceTarget {
  self.identity.target
}

///|
pub fn ApprovedPagedAttentionReadOnlyModule::operation_id(
  self : ApprovedPagedAttentionReadOnlyModule,
) -> Int {
  self.identity.operation
}

///|
pub fn ApprovedPagedAttentionReadOnlyModule::activation_ref(
  self : ApprovedPagedAttentionReadOnlyModule,
) -> Int {
  self.identity.activation
}

///|
pub fn ApprovedPagedAttentionReadOnlyModule::input_row_width(
  self : ApprovedPagedAttentionReadOnlyModule,
) -> Int {
  self.identity.input_row_width
}

///|
pub fn ApprovedPagedAttentionReadOnlyModule::profile_shape(
  self : ApprovedPagedAttentionReadOnlyModule,
) -> (Int, Int, Int) {
  (
    self.identity.profile_rows,
    self.identity.profile_tokens,
    self.identity.profile_page_entries,
  )
}

///|
pub fn ApprovedPagedAttentionReadOnlyModule::profile_id(
  self : ApprovedPagedAttentionReadOnlyModule,
) -> Int {
  self.identity.profile_id
}

///|
pub fn ApprovedPagedAttentionReadOnlyModule::function_symbol(
  self : ApprovedPagedAttentionReadOnlyModule,
) -> String {
  self.identity.symbol
}

///|
pub fn ApprovedPagedAttentionReadOnlyModule::kernel_abi_name(
  self : ApprovedPagedAttentionReadOnlyModule,
) -> String {
  self.identity.abi
}

///|
pub fn ApprovedPagedAttentionReadOnlyModule::launch_dimensions(
  self : ApprovedPagedAttentionReadOnlyModule,
) -> @launch_contract.AotLaunchDimensions {
  self.launch
}

///|
pub fn ApprovedPagedAttentionReadOnlyModule::module_bytes(
  self : ApprovedPagedAttentionReadOnlyModule,
) -> Bytes {
  self.module_bytes
}

///|
pub fn ApprovedPagedAttentionReadOnlyModule::dispatch_canary_per_token(
  self : ApprovedPagedAttentionReadOnlyModule,
) -> Int {
  self.identity.dispatch_canary
}

///|
pub fn ApprovedPagedAttentionReadOnlyModule::dispatch_canary_cell_count(
  self : ApprovedPagedAttentionReadOnlyModule,
) -> Int {
  self.identity.dispatch_canary_cells
}

///|
pub fn ApprovedPagedAttentionReadOnlyModule::tokens_per_page(
  self : ApprovedPagedAttentionReadOnlyModule,
) -> Int {
  self.identity.tokens_per_page
}

///|
pub fn ApprovedPagedAttentionReadOnlyModule::page_stride_bytes(
  self : ApprovedPagedAttentionReadOnlyModule,
) -> Int64 {
  self.identity.page_stride_bytes
}

///|
pub fn ApprovedPagedAttentionReadOnlyModule::physical_page_capacity(
  self : ApprovedPagedAttentionReadOnlyModule,
) -> Int64 {
  self.identity.physical_page_capacity
}

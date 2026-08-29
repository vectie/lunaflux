///|
pub fn QualifiedPagedKvWriteArtifact::candidate_digest(
  self : QualifiedPagedKvWriteArtifact,
) -> @luna_cuda_paged_kv_write_aot.PagedKvWriteDigest {
  self.candidate.candidate_digest()
}

///|
pub fn QualifiedPagedKvWriteArtifact::source_digest(
  self : QualifiedPagedKvWriteArtifact,
) -> @luna_cuda_paged_kv_write_aot.PagedKvWriteDigest {
  self.candidate.source_digest()
}

///|
pub fn QualifiedPagedKvWriteArtifact::recipe_digest(
  self : QualifiedPagedKvWriteArtifact,
) -> @luna_cuda_paged_kv_write_aot.PagedKvWriteDigest {
  self.candidate.recipe_digest()
}

///|
pub fn QualifiedPagedKvWriteArtifact::compiled_binding_digest(
  self : QualifiedPagedKvWriteArtifact,
) -> @luna_cuda_paged_kv_write_aot.PagedKvWriteDigest {
  self.compiled.binding_digest()
}

///|
pub fn QualifiedPagedKvWriteArtifact::module_digest(
  self : QualifiedPagedKvWriteArtifact,
) -> @luna_cuda_paged_kv_write_aot.PagedKvWriteDigest {
  self.compiled.module_digest()
}

///|
pub fn QualifiedPagedKvWriteArtifact::compile_receipt_digest(
  self : QualifiedPagedKvWriteArtifact,
) -> @luna_cuda_paged_kv_write_aot.PagedKvWriteDigest {
  self.compiled.compile_receipt_digest()
}

///|
pub fn QualifiedPagedKvWriteArtifact::evidence_digest(
  self : QualifiedPagedKvWriteArtifact,
) -> @luna_cuda_paged_kv_write_aot.PagedKvWriteDigest {
  self.evidence_digest
}

///|
pub fn QualifiedPagedKvWriteArtifact::approved_policy_digest(
  self : QualifiedPagedKvWriteArtifact,
) -> @luna_cuda_paged_kv_write_aot.PagedKvWriteDigest {
  self.approved_policy_digest
}

///|
pub fn QualifiedPagedKvWriteArtifact::device_uuid(
  self : QualifiedPagedKvWriteArtifact,
) -> String {
  self.device_uuid
}

///|
pub fn QualifiedPagedKvWriteArtifact::device_pci(
  self : QualifiedPagedKvWriteArtifact,
) -> String {
  self.device_pci
}

///|
pub fn QualifiedPagedKvWriteArtifact::files_manifest_digest(
  self : QualifiedPagedKvWriteArtifact,
) -> @luna_cuda_paged_kv_write_aot.PagedKvWriteDigest {
  self.files_manifest_digest
}

///|
pub fn QualifiedPagedKvWriteArtifact::outer_seal_digest(
  self : QualifiedPagedKvWriteArtifact,
) -> @luna_cuda_paged_kv_write_aot.PagedKvWriteDigest {
  self.outer_seal_digest
}

///|
pub fn QualifiedPagedKvWriteArtifact::model_content_digest_sha256(
  self : QualifiedPagedKvWriteArtifact,
) -> String {
  self.candidate.model_content_digest_sha256()
}

///|
pub fn QualifiedPagedKvWriteArtifact::model_plan_digest_sha256(
  self : QualifiedPagedKvWriteArtifact,
) -> String {
  self.candidate.model_plan_digest_sha256()
}

///|
pub fn QualifiedPagedKvWriteArtifact::operation_id(
  self : QualifiedPagedKvWriteArtifact,
) -> @plan.OperationId {
  self.candidate.operation_id()
}

///|
pub fn QualifiedPagedKvWriteArtifact::positioned_qkv_activation(
  self : QualifiedPagedKvWriteArtifact,
) -> @plan.ActivationRef {
  self.candidate.positioned_qkv_activation()
}

///|
pub fn QualifiedPagedKvWriteArtifact::query_heads(
  self : QualifiedPagedKvWriteArtifact,
) -> Int {
  self.candidate.query_heads()
}

///|
pub fn QualifiedPagedKvWriteArtifact::kv_heads(
  self : QualifiedPagedKvWriteArtifact,
) -> Int {
  self.candidate.kv_heads()
}

///|
pub fn QualifiedPagedKvWriteArtifact::head_dimension(
  self : QualifiedPagedKvWriteArtifact,
) -> Int {
  self.candidate.head_dimension()
}

///|
pub fn QualifiedPagedKvWriteArtifact::positioned_qkv_row_width(
  self : QualifiedPagedKvWriteArtifact,
) -> Int {
  (self.query_heads() + 2 * self.kv_heads()) * self.head_dimension()
}

///|
pub fn QualifiedPagedKvWriteArtifact::target(
  self : QualifiedPagedKvWriteArtifact,
) -> @catalog.DeviceTarget {
  self.candidate.target()
}

///|
pub fn QualifiedPagedKvWriteArtifact::profile(
  self : QualifiedPagedKvWriteArtifact,
) -> @launch_contract.PagedKernelProfile {
  self.candidate.profile()
}

///|
pub fn QualifiedPagedKvWriteArtifact::tokens_per_page(
  self : QualifiedPagedKvWriteArtifact,
) -> Int {
  self.candidate.tokens_per_page()
}

///|
pub fn QualifiedPagedKvWriteArtifact::total_page_count(
  self : QualifiedPagedKvWriteArtifact,
) -> Int {
  self.candidate.total_page_count()
}

///|
pub fn QualifiedPagedKvWriteArtifact::page_stride_bytes(
  self : QualifiedPagedKvWriteArtifact,
) -> Int64 {
  self.candidate.page_stride_bytes()
}

///|
pub fn QualifiedPagedKvWriteArtifact::toolchain_digest_sha256(
  self : QualifiedPagedKvWriteArtifact,
) -> String {
  self.candidate.compiler_policy().identity().toolchain_digest_sha256()
}

///|
pub fn QualifiedPagedKvWriteArtifact::driver_identity_sha256(
  self : QualifiedPagedKvWriteArtifact,
) -> String {
  self.compiled.driver_identity_sha256()
}

///|
pub fn QualifiedPagedKvWriteArtifact::function_symbol(
  self : QualifiedPagedKvWriteArtifact,
) -> String {
  self.candidate.function_symbol()
}

///|
pub fn QualifiedPagedKvWriteArtifact::kernel_abi(
  self : QualifiedPagedKvWriteArtifact,
) -> @luna_cuda_paged_kv_write_aot.PagedKvWriteKernelAbi {
  self.candidate.kernel_abi()
}

///|
pub fn QualifiedPagedKvWriteArtifact::launch_dimensions(
  self : QualifiedPagedKvWriteArtifact,
) -> @launch_contract.AotLaunchDimensions {
  self.candidate.launch_dimensions()
}

///|
pub fn QualifiedPagedKvWriteArtifact::dispatch_canary_per_token(
  self : QualifiedPagedKvWriteArtifact,
) -> UInt {
  self.candidate.dispatch_canary_per_token()
}

///|
pub fn QualifiedPagedKvWriteArtifact::fallback_source_digest_sha256(
  self : QualifiedPagedKvWriteArtifact,
) -> String {
  self.candidate.fallback_source_digest_sha256()
}

///|
pub fn QualifiedPagedKvWriteArtifact::fallback_recipe_digest_sha256(
  self : QualifiedPagedKvWriteArtifact,
) -> String {
  self.candidate.fallback_recipe_digest_sha256()
}

///|
pub fn QualifiedPagedKvWriteArtifact::manifest_bindable(
  _self : QualifiedPagedKvWriteArtifact,
) -> Bool {
  false
}

///|
pub fn QualifiedPagedKvWriteArtifact::promotion_authority(
  _self : QualifiedPagedKvWriteArtifact,
) -> Bool {
  false
}

///|
pub fn QualifiedPagedKvWriteArtifact::runtime_dispatch_authority(
  _self : QualifiedPagedKvWriteArtifact,
) -> Bool {
  false
}

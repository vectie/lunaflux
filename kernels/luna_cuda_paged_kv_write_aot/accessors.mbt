///|
pub fn PagedKvWriteCudaCandidate::operation_id(
  self : PagedKvWriteCudaCandidate,
) -> @plan.OperationId {
  self.attention_operation
}

///|
pub fn PagedKvWriteCudaCandidate::model_content_digest_sha256(
  self : PagedKvWriteCudaCandidate,
) -> String {
  self.model_content
}

///|
pub fn PagedKvWriteCudaCandidate::model_plan_digest_sha256(
  self : PagedKvWriteCudaCandidate,
) -> String {
  self.model_plan
}

///|
pub fn PagedKvWriteCudaCandidate::tokens_per_page(
  self : PagedKvWriteCudaCandidate,
) -> Int {
  self.tokens_per_page
}

///|
pub fn PagedKvWriteCudaCandidate::total_page_count(
  self : PagedKvWriteCudaCandidate,
) -> Int {
  self.total_page_count
}

///|
pub fn PagedKvWriteCudaCandidate::page_stride_bytes(
  self : PagedKvWriteCudaCandidate,
) -> Int64 {
  self.page_stride_bytes
}

///|
pub fn PagedKvWriteCudaCandidate::kv_width(
  self : PagedKvWriteCudaCandidate,
) -> Int {
  self.kv_width
}

///|
pub fn PagedKvWriteCudaCandidate::query_heads(
  self : PagedKvWriteCudaCandidate,
) -> Int {
  self.query_heads
}

///|
pub fn PagedKvWriteCudaCandidate::kv_heads(
  self : PagedKvWriteCudaCandidate,
) -> Int {
  self.kv_heads
}

///|
pub fn PagedKvWriteCudaCandidate::head_dimension(
  self : PagedKvWriteCudaCandidate,
) -> Int {
  self.head_dimension
}

///|
pub fn PagedKvWriteCudaCandidate::positioned_qkv_activation(
  self : PagedKvWriteCudaCandidate,
) -> @plan.ActivationRef {
  self.positioned_qkv
}

///|
pub fn PagedKvWriteCudaCandidate::target(
  self : PagedKvWriteCudaCandidate,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn PagedKvWriteCudaCandidate::profile(
  self : PagedKvWriteCudaCandidate,
) -> @launch_contract.PagedKernelProfile {
  self.profile
}

///|
pub fn PagedKvWriteCudaCandidate::compiler_policy(
  self : PagedKvWriteCudaCandidate,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn PagedKvWriteCudaCandidate::operands(
  self : PagedKvWriteCudaCandidate,
) -> ReadOnlyArray[@launch_contract.AotOperand] {
  self.operands
}

///|
pub fn PagedKvWriteCudaCandidate::launch_dimensions(
  self : PagedKvWriteCudaCandidate,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn PagedKvWriteCudaCandidate::function_symbol(
  self : PagedKvWriteCudaCandidate,
) -> String {
  self.symbol
}

///|
pub fn PagedKvWriteCudaCandidate::kernel_abi(
  self : PagedKvWriteCudaCandidate,
) -> PagedKvWriteKernelAbi {
  self.abi
}

///|
pub fn PagedKvWriteCudaCandidate::dispatch_canary_per_token(
  self : PagedKvWriteCudaCandidate,
) -> UInt {
  self.dispatch_canary
}

///|
pub fn PagedKvWriteCudaCandidate::source_bytes(
  self : PagedKvWriteCudaCandidate,
) -> Bytes {
  self.source
}

///|
pub fn PagedKvWriteCudaCandidate::source_digest(
  self : PagedKvWriteCudaCandidate,
) -> PagedKvWriteDigest {
  self.source_digest
}

///|
pub fn PagedKvWriteCudaCandidate::recipe_bytes(
  self : PagedKvWriteCudaCandidate,
) -> Bytes {
  self.recipe
}

///|
pub fn PagedKvWriteCudaCandidate::recipe_digest(
  self : PagedKvWriteCudaCandidate,
) -> PagedKvWriteDigest {
  self.recipe_digest
}

///|
pub fn PagedKvWriteCudaCandidate::candidate_digest(
  self : PagedKvWriteCudaCandidate,
) -> PagedKvWriteDigest {
  self.candidate_digest
}

///|
pub fn PagedKvWriteCudaCandidate::fallback_source_digest_sha256(
  self : PagedKvWriteCudaCandidate,
) -> String {
  self.fallback_source
}

///|
pub fn PagedKvWriteCudaCandidate::fallback_recipe_digest_sha256(
  self : PagedKvWriteCudaCandidate,
) -> String {
  self.fallback_recipe
}

///|
pub fn PagedKvWriteCudaCandidate::manifest_bindable(
  _self : PagedKvWriteCudaCandidate,
) -> Bool {
  false
}

///|
pub fn PagedKvWriteCudaCandidate::promotion_authority(
  _self : PagedKvWriteCudaCandidate,
) -> Bool {
  false
}

///|
pub fn PagedKvWriteCompiledArtifact::candidate_digest(
  self : PagedKvWriteCompiledArtifact,
) -> PagedKvWriteDigest {
  self.candidate
}

///|
pub fn PagedKvWriteCompiledArtifact::source_digest(
  self : PagedKvWriteCompiledArtifact,
) -> PagedKvWriteDigest {
  self.source
}

///|
pub fn PagedKvWriteCompiledArtifact::recipe_digest(
  self : PagedKvWriteCompiledArtifact,
) -> PagedKvWriteDigest {
  self.recipe
}

///|
pub fn PagedKvWriteCompiledArtifact::target(
  self : PagedKvWriteCompiledArtifact,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn PagedKvWriteCompiledArtifact::function_symbol(
  self : PagedKvWriteCompiledArtifact,
) -> String {
  self.symbol
}

///|
pub fn PagedKvWriteCompiledArtifact::kernel_abi(
  self : PagedKvWriteCompiledArtifact,
) -> PagedKvWriteKernelAbi {
  self.abi
}

///|
pub fn PagedKvWriteCompiledArtifact::module_bytes(
  self : PagedKvWriteCompiledArtifact,
) -> Bytes {
  self.module_bytes
}

///|
pub fn PagedKvWriteCompiledArtifact::module_digest(
  self : PagedKvWriteCompiledArtifact,
) -> PagedKvWriteDigest {
  self.module_digest
}

///|
pub fn PagedKvWriteCompiledArtifact::binding_digest(
  self : PagedKvWriteCompiledArtifact,
) -> PagedKvWriteDigest {
  self.binding_digest
}

///|
pub fn PagedKvWriteCompiledArtifact::binding_record_bytes(
  self : PagedKvWriteCompiledArtifact,
) -> Bytes {
  self.binding
}

///|
pub fn PagedKvWriteCompiledArtifact::compile_receipt_bytes(
  self : PagedKvWriteCompiledArtifact,
) -> Bytes {
  self.receipt
}

///|
pub fn PagedKvWriteCompiledArtifact::compile_receipt_digest(
  self : PagedKvWriteCompiledArtifact,
) -> PagedKvWriteDigest {
  self.receipt_digest
}

///|
pub fn PagedKvWriteCompiledArtifact::driver_identity_sha256(
  self : PagedKvWriteCompiledArtifact,
) -> String {
  self.driver_identity
}

///|
pub fn PagedKvWriteCompiledArtifact::manifest_bindable(
  _self : PagedKvWriteCompiledArtifact,
) -> Bool {
  false
}

///|
pub fn PagedKvWriteCompiledArtifact::promotion_authority(
  _self : PagedKvWriteCompiledArtifact,
) -> Bool {
  false
}

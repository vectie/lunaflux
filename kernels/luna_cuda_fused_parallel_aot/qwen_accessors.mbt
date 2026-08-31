///|
pub fn FusedQwenQkvQkNormRopeKvWriteCudaCandidate::semantic_plan(
  self : FusedQwenQkvQkNormRopeKvWriteCudaCandidate,
) -> @luna_fusion_plan.QkvQkNormPositionedRopePagedKvWritePlan {
  self.semantic
}

///|
pub fn FusedQwenQkvQkNormRopeKvWriteCudaCandidate::target(
  self : FusedQwenQkvQkNormRopeKvWriteCudaCandidate,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn FusedQwenQkvQkNormRopeKvWriteCudaCandidate::profile(
  self : FusedQwenQkvQkNormRopeKvWriteCudaCandidate,
) -> @launch_contract.PagedKernelProfile {
  self.profile
}

///|
pub fn FusedQwenQkvQkNormRopeKvWriteCudaCandidate::compiler_policy(
  self : FusedQwenQkvQkNormRopeKvWriteCudaCandidate,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn FusedQwenQkvQkNormRopeKvWriteCudaCandidate::kv_layout(
  self : FusedQwenQkvQkNormRopeKvWriteCudaCandidate,
) -> @device_layout.DeviceKvLayout {
  self.layout
}

///|
pub fn FusedQwenQkvQkNormRopeKvWriteCudaCandidate::numerical_policy(
  self : FusedQwenQkvQkNormRopeKvWriteCudaCandidate,
) -> FusedParallelNumericalPolicy {
  self.numerical_policy
}

///|
pub fn FusedQwenQkvQkNormRopeKvWriteCudaCandidate::fallback_source_digests(
  self : FusedQwenQkvQkNormRopeKvWriteCudaCandidate,
) -> (String, String, String, String) {
  (
    self.qkv_fallback.source_digest().as_hex(),
    self.qk_norm_fallback.source_digest().as_hex(),
    self.rope_fallback.source_digest().as_hex(),
    self.kv_write_fallback.source_digest().as_hex(),
  )
}

///|
pub fn FusedQwenQkvQkNormRopeKvWriteCudaCandidate::launch_dimensions(
  self : FusedQwenQkvQkNormRopeKvWriteCudaCandidate,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn FusedQwenQkvQkNormRopeKvWriteCudaCandidate::function_symbol(
  self : FusedQwenQkvQkNormRopeKvWriteCudaCandidate,
) -> String {
  self.function_symbol
}

///|
pub fn FusedQwenQkvQkNormRopeKvWriteCudaCandidate::source_bytes(
  self : FusedQwenQkvQkNormRopeKvWriteCudaCandidate,
) -> Bytes {
  self.source
}

///|
pub fn FusedQwenQkvQkNormRopeKvWriteCudaCandidate::source_digest(
  self : FusedQwenQkvQkNormRopeKvWriteCudaCandidate,
) -> FusedParallelDigest {
  self.source_digest
}

///|
pub fn FusedQwenQkvQkNormRopeKvWriteCudaCandidate::recipe_bytes(
  self : FusedQwenQkvQkNormRopeKvWriteCudaCandidate,
) -> Bytes {
  self.recipe
}

///|
pub fn FusedQwenQkvQkNormRopeKvWriteCudaCandidate::recipe_digest(
  self : FusedQwenQkvQkNormRopeKvWriteCudaCandidate,
) -> FusedParallelDigest {
  self.recipe_digest
}

///|
pub fn FusedQwenQkvQkNormRopeKvWriteCudaCandidate::candidate_digest(
  self : FusedQwenQkvQkNormRopeKvWriteCudaCandidate,
) -> FusedParallelDigest {
  self.candidate_digest
}

///|
pub fn FusedQwenQkvQkNormRopeKvWriteCudaCandidate::dispatch_canary(
  self : FusedQwenQkvQkNormRopeKvWriteCudaCandidate,
) -> UInt {
  self.dispatch_canary
}

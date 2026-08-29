///|
pub fn FusedQkvRopeKvWriteCudaCandidate::target(
  self : FusedQkvRopeKvWriteCudaCandidate,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn FusedQkvRopeKvWriteCudaCandidate::profile(
  self : FusedQkvRopeKvWriteCudaCandidate,
) -> @launch_contract.PagedKernelProfile {
  self.profile
}

///|
pub fn FusedQkvRopeKvWriteCudaCandidate::compiler_policy(
  self : FusedQkvRopeKvWriteCudaCandidate,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn FusedQkvRopeKvWriteCudaCandidate::launch_dimensions(
  self : FusedQkvRopeKvWriteCudaCandidate,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn FusedQkvRopeKvWriteCudaCandidate::numerical_policy(
  self : FusedQkvRopeKvWriteCudaCandidate,
) -> FusedParallelNumericalPolicy {
  self.numerical_policy
}

///|
pub fn FusedQkvRopeKvWriteCudaCandidate::function_symbol(
  self : FusedQkvRopeKvWriteCudaCandidate,
) -> String {
  self.function_symbol
}

///|
pub fn FusedQkvRopeKvWriteCudaCandidate::dispatch_canary(
  self : FusedQkvRopeKvWriteCudaCandidate,
) -> UInt {
  self.dispatch_canary
}

///|
pub fn FusedQkvRopeKvWriteCudaCandidate::source_bytes(
  self : FusedQkvRopeKvWriteCudaCandidate,
) -> Bytes {
  self.source
}

///|
pub fn FusedQkvRopeKvWriteCudaCandidate::source_digest(
  self : FusedQkvRopeKvWriteCudaCandidate,
) -> FusedParallelDigest {
  self.source_digest
}

///|
pub fn FusedQkvRopeKvWriteCudaCandidate::recipe_bytes(
  self : FusedQkvRopeKvWriteCudaCandidate,
) -> Bytes {
  self.recipe
}

///|
pub fn FusedQkvRopeKvWriteCudaCandidate::recipe_digest(
  self : FusedQkvRopeKvWriteCudaCandidate,
) -> FusedParallelDigest {
  self.recipe_digest
}

///|
pub fn FusedQkvRopeKvWriteCudaCandidate::candidate_digest(
  self : FusedQkvRopeKvWriteCudaCandidate,
) -> FusedParallelDigest {
  self.candidate_digest
}

///|
pub fn FusedQkvRopeKvWriteCudaCandidate::manifest_bindable(
  _self : FusedQkvRopeKvWriteCudaCandidate,
) -> Bool {
  false
}

///|
pub fn FusedQkvRopeKvWriteCudaCandidate::qkv_fallback(
  self : FusedQkvRopeKvWriteCudaCandidate,
) -> @luna_cuda_projection_aot.ProjectionCudaLowering {
  self.qkv_fallback
}

///|
pub fn FusedQkvRopeKvWriteCudaCandidate::rope_fallback(
  self : FusedQkvRopeKvWriteCudaCandidate,
) -> @luna_cuda_pointwise_aot.PointwiseCudaLowering {
  self.rope_fallback
}

///|
pub fn FusedQkvRopeKvWriteCudaCandidate::kv_write_fallback(
  self : FusedQkvRopeKvWriteCudaCandidate,
) -> @luna_cuda_pointwise_aot.PointwiseCudaLowering {
  self.kv_write_fallback
}

///|
pub fn FusedQkvRopeKvWriteCudaCandidate::head_dimension(
  self : FusedQkvRopeKvWriteCudaCandidate,
) -> Int {
  self.parameters.head_dimension
}

///|
pub fn FusedQkvRopeKvWriteCudaCandidate::tokens_per_page(
  self : FusedQkvRopeKvWriteCudaCandidate,
) -> Int {
  self.parameters.tokens_per_page
}

///|
pub fn FusedQkvRopeKvWriteProductionCudaCandidate::qualification_candidate_digest(
  self : FusedQkvRopeKvWriteProductionCudaCandidate,
) -> FusedParallelDigest {
  self.qualification.candidate_digest
}

///|
pub fn FusedQkvRopeKvWriteProductionCudaCandidate::function_symbol(
  self : FusedQkvRopeKvWriteProductionCudaCandidate,
) -> String {
  self.function_symbol
}

///|
pub fn FusedQkvRopeKvWriteProductionCudaCandidate::target(
  self : FusedQkvRopeKvWriteProductionCudaCandidate,
) -> @catalog.DeviceTarget {
  self.qualification.target
}

///|
pub fn FusedQkvRopeKvWriteProductionCudaCandidate::compiler_policy(
  self : FusedQkvRopeKvWriteProductionCudaCandidate,
) -> @luna_specialization.LunaCompilerPolicy {
  self.qualification.compiler
}

///|
pub fn FusedQkvRopeKvWriteProductionCudaCandidate::source_bytes(
  self : FusedQkvRopeKvWriteProductionCudaCandidate,
) -> Bytes {
  self.source
}

///|
pub fn FusedQkvRopeKvWriteProductionCudaCandidate::source_digest(
  self : FusedQkvRopeKvWriteProductionCudaCandidate,
) -> FusedParallelDigest {
  self.source_digest
}

///|
pub fn FusedQkvRopeKvWriteProductionCudaCandidate::recipe_bytes(
  self : FusedQkvRopeKvWriteProductionCudaCandidate,
) -> Bytes {
  self.recipe
}

///|
pub fn FusedQkvRopeKvWriteProductionCudaCandidate::recipe_digest(
  self : FusedQkvRopeKvWriteProductionCudaCandidate,
) -> FusedParallelDigest {
  self.recipe_digest
}

///|
pub fn FusedQkvRopeKvWriteProductionCudaCandidate::candidate_digest(
  self : FusedQkvRopeKvWriteProductionCudaCandidate,
) -> FusedParallelDigest {
  self.candidate_digest
}

///|
pub fn FusedQkvRopeKvWriteProductionCudaCandidate::manifest_bindable(
  _self : FusedQkvRopeKvWriteProductionCudaCandidate,
) -> Bool {
  false
}

///|
pub fn FusedResidualRmsNormCudaCandidate::target(
  self : FusedResidualRmsNormCudaCandidate,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn FusedResidualRmsNormCudaCandidate::profile(
  self : FusedResidualRmsNormCudaCandidate,
) -> @launch_contract.PagedKernelProfile {
  self.profile
}

///|
pub fn FusedResidualRmsNormCudaCandidate::compiler_policy(
  self : FusedResidualRmsNormCudaCandidate,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn FusedResidualRmsNormCudaCandidate::launch_dimensions(
  self : FusedResidualRmsNormCudaCandidate,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn FusedResidualRmsNormCudaCandidate::numerical_policy(
  self : FusedResidualRmsNormCudaCandidate,
) -> FusedParallelNumericalPolicy {
  self.numerical_policy
}

///|
pub fn FusedResidualRmsNormCudaCandidate::function_symbol(
  self : FusedResidualRmsNormCudaCandidate,
) -> String {
  self.function_symbol
}

///|
pub fn FusedResidualRmsNormCudaCandidate::dispatch_canary(
  self : FusedResidualRmsNormCudaCandidate,
) -> UInt {
  self.dispatch_canary
}

///|
pub fn FusedResidualRmsNormCudaCandidate::source_bytes(
  self : FusedResidualRmsNormCudaCandidate,
) -> Bytes {
  self.source
}

///|
pub fn FusedResidualRmsNormCudaCandidate::source_digest(
  self : FusedResidualRmsNormCudaCandidate,
) -> FusedParallelDigest {
  self.source_digest
}

///|
pub fn FusedResidualRmsNormCudaCandidate::recipe_bytes(
  self : FusedResidualRmsNormCudaCandidate,
) -> Bytes {
  self.recipe
}

///|
pub fn FusedResidualRmsNormCudaCandidate::recipe_digest(
  self : FusedResidualRmsNormCudaCandidate,
) -> FusedParallelDigest {
  self.recipe_digest
}

///|
pub fn FusedResidualRmsNormCudaCandidate::candidate_digest(
  self : FusedResidualRmsNormCudaCandidate,
) -> FusedParallelDigest {
  self.candidate_digest
}

///|
pub fn FusedResidualRmsNormCudaCandidate::manifest_bindable(
  _self : FusedResidualRmsNormCudaCandidate,
) -> Bool {
  false
}

///|
pub fn FusedResidualRmsNormCudaCandidate::residual_fallback(
  self : FusedResidualRmsNormCudaCandidate,
) -> @luna_cuda_pointwise_aot.PointwiseCudaLowering {
  self.residual_fallback
}

///|
pub fn FusedResidualRmsNormCudaCandidate::norm_fallback(
  self : FusedResidualRmsNormCudaCandidate,
) -> @luna_cuda_pointwise_aot.PointwiseCudaLowering {
  self.norm_fallback
}

///|
pub fn FusedResidualRmsNormCudaCandidate::width(
  self : FusedResidualRmsNormCudaCandidate,
) -> Int {
  self.parameters.width
}

///|
pub fn FusedResidualRmsNormProductionCudaCandidate::qualification_candidate_digest(
  self : FusedResidualRmsNormProductionCudaCandidate,
) -> FusedParallelDigest {
  self.qualification.candidate_digest
}

///|
pub fn FusedResidualRmsNormProductionCudaCandidate::target(
  self : FusedResidualRmsNormProductionCudaCandidate,
) -> @catalog.DeviceTarget {
  self.qualification.target
}

///|
pub fn FusedResidualRmsNormProductionCudaCandidate::profile(
  self : FusedResidualRmsNormProductionCudaCandidate,
) -> @launch_contract.PagedKernelProfile {
  self.qualification.profile
}

///|
pub fn FusedResidualRmsNormProductionCudaCandidate::compiler_policy(
  self : FusedResidualRmsNormProductionCudaCandidate,
) -> @luna_specialization.LunaCompilerPolicy {
  self.qualification.compiler
}

///|
pub fn FusedResidualRmsNormProductionCudaCandidate::launch_dimensions(
  self : FusedResidualRmsNormProductionCudaCandidate,
) -> @launch_contract.AotLaunchDimensions {
  self.qualification.dimensions
}

///|
pub fn FusedResidualRmsNormProductionCudaCandidate::numerical_policy(
  self : FusedResidualRmsNormProductionCudaCandidate,
) -> FusedParallelNumericalPolicy {
  self.qualification.numerical_policy
}

///|
pub fn FusedResidualRmsNormProductionCudaCandidate::function_symbol(
  self : FusedResidualRmsNormProductionCudaCandidate,
) -> String {
  self.function_symbol
}

///|
pub fn FusedResidualRmsNormProductionCudaCandidate::source_bytes(
  self : FusedResidualRmsNormProductionCudaCandidate,
) -> Bytes {
  self.source
}

///|
pub fn FusedResidualRmsNormProductionCudaCandidate::source_digest(
  self : FusedResidualRmsNormProductionCudaCandidate,
) -> FusedParallelDigest {
  self.source_digest
}

///|
pub fn FusedResidualRmsNormProductionCudaCandidate::recipe_bytes(
  self : FusedResidualRmsNormProductionCudaCandidate,
) -> Bytes {
  self.recipe
}

///|
pub fn FusedResidualRmsNormProductionCudaCandidate::recipe_digest(
  self : FusedResidualRmsNormProductionCudaCandidate,
) -> FusedParallelDigest {
  self.recipe_digest
}

///|
pub fn FusedResidualRmsNormProductionCudaCandidate::candidate_digest(
  self : FusedResidualRmsNormProductionCudaCandidate,
) -> FusedParallelDigest {
  self.candidate_digest
}

///|
pub fn FusedResidualRmsNormProductionCudaCandidate::manifest_bindable(
  _self : FusedResidualRmsNormProductionCudaCandidate,
) -> Bool {
  false
}

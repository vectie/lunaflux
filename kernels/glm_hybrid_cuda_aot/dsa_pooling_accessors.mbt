///|
pub fn GlmHybridDsaPoolingCudaCandidate::family(
  self : Self,
) -> GlmHybridDsaPoolingAotFamily {
  self.family
}

///|
pub fn GlmHybridDsaPoolingCudaCandidate::numerical_contract(
  self : Self,
) -> @glm_hybrid_dsa_pooling_cuda_source.GlmHybridDsaPoolingNumericalContract {
  self.numerical_contract
}

///|
pub fn GlmHybridDsaPoolingCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn GlmHybridDsaPoolingCudaCandidate::requirements_digest(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn GlmHybridDsaPoolingCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn GlmHybridDsaPoolingCudaCandidate::operation(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation {
  self.operation
}

///|
pub fn GlmHybridDsaPoolingCudaCandidate::capability(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridAotCapability {
  self.capability
}

///|
pub fn GlmHybridDsaPoolingCudaCandidate::head_dimension(self : Self) -> Int {
  self.head_dimension
}

///|
pub fn GlmHybridDsaPoolingCudaCandidate::pool_size(self : Self) -> Int {
  self.pool_size
}

///|
pub fn GlmHybridDsaPoolingCudaCandidate::maximum_tokens(self : Self) -> Int {
  self.maximum_tokens
}

///|
pub fn GlmHybridDsaPoolingCudaCandidate::pool_capacity(self : Self) -> Int {
  self.pool_capacity
}

///|
pub fn GlmHybridDsaPoolingCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn GlmHybridDsaPoolingCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn GlmHybridDsaPoolingCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[@glm_hybrid_artifact_admission.GlmHybridLaunchOperand] {
  self.operands
}

///|
pub fn GlmHybridDsaPoolingCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn GlmHybridDsaPoolingCudaCandidate::function_symbol(self : Self) -> String {
  self.function_symbol
}

///|
pub fn GlmHybridDsaPoolingCudaCandidate::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn GlmHybridDsaPoolingCudaCandidate::source_digest(
  self : Self,
) -> GlmHybridCudaSourceDigest {
  self.source_digest
}

///|
pub fn GlmHybridDsaPoolingCudaCandidate::recipe_bytes(self : Self) -> Bytes {
  self.recipe
}

///|
pub fn GlmHybridDsaPoolingCudaCandidate::recipe_digest(
  self : Self,
) -> GlmHybridCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn GlmHybridDsaPoolingCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

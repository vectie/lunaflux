///|
pub fn GlmHybridDsaIndexCudaCandidate::family(
  self : Self,
) -> GlmHybridDsaIndexAotFamily {
  self.family
}

///|
pub fn GlmHybridDsaIndexCudaCandidate::numerical_contract(
  self : Self,
) -> @glm_hybrid_dsa_index_cuda_source.GlmHybridDsaIndexNumericalContract {
  self.numerical_contract
}

///|
pub fn GlmHybridDsaIndexCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn GlmHybridDsaIndexCudaCandidate::requirements_digest(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn GlmHybridDsaIndexCudaCandidate::requirement_ordinal(self : Self) -> Int {
  self.requirement_ordinal
}

///|
pub fn GlmHybridDsaIndexCudaCandidate::operation(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation {
  self.operation
}

///|
pub fn GlmHybridDsaIndexCudaCandidate::capability(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridAotCapability {
  self.capability
}

///|
pub fn GlmHybridDsaIndexCudaCandidate::head_count(self : Self) -> Int {
  self.head_count
}

///|
pub fn GlmHybridDsaIndexCudaCandidate::head_dimension(self : Self) -> Int {
  self.head_dimension
}

///|
pub fn GlmHybridDsaIndexCudaCandidate::top_k(self : Self) -> Int {
  self.top_k
}

///|
pub fn GlmHybridDsaIndexCudaCandidate::pool_size(self : Self) -> Int {
  self.pool_size
}

///|
pub fn GlmHybridDsaIndexCudaCandidate::maximum_tokens(self : Self) -> Int {
  self.maximum_tokens
}

///|
pub fn GlmHybridDsaIndexCudaCandidate::candidate_capacity(self : Self) -> Int {
  self.candidate_capacity
}

///|
pub fn GlmHybridDsaIndexCudaCandidate::output_width(self : Self) -> Int {
  self.output_width
}

///|
pub fn GlmHybridDsaIndexCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn GlmHybridDsaIndexCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn GlmHybridDsaIndexCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[@glm_hybrid_artifact_admission.GlmHybridLaunchOperand] {
  self.operands
}

///|
pub fn GlmHybridDsaIndexCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn GlmHybridDsaIndexCudaCandidate::function_symbol(self : Self) -> String {
  self.function_symbol
}

///|
pub fn GlmHybridDsaIndexCudaCandidate::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn GlmHybridDsaIndexCudaCandidate::source_digest(
  self : Self,
) -> GlmHybridCudaSourceDigest {
  self.source_digest
}

///|
pub fn GlmHybridDsaIndexCudaCandidate::recipe_bytes(self : Self) -> Bytes {
  self.recipe
}

///|
pub fn GlmHybridDsaIndexCudaCandidate::recipe_digest(
  self : Self,
) -> GlmHybridCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn GlmHybridDsaIndexCudaCandidate::manifest_bindable(_self : Self) -> Bool {
  false
}

///|
pub fn GlmHybridKdaProjectionCudaCandidate::family(
  self : Self,
) -> GlmHybridKdaProjectionAotFamily {
  self.family
}

///|
pub fn GlmHybridKdaProjectionCudaCandidate::numerical_contract(
  self : Self,
) -> @recurrent_attention_projection_cuda_source.RecurrentAttentionProjectionNumericalContract {
  self.numerical_contract
}

///|
pub fn GlmHybridKdaProjectionCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn GlmHybridKdaProjectionCudaCandidate::requirements_digest(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn GlmHybridKdaProjectionCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn GlmHybridKdaProjectionCudaCandidate::operation(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation {
  self.operation
}

///|
pub fn GlmHybridKdaProjectionCudaCandidate::capability(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridAotCapability {
  self.capability
}

///|
pub fn GlmHybridKdaProjectionCudaCandidate::hidden_width(self : Self) -> Int {
  self.hidden_width
}

///|
pub fn GlmHybridKdaProjectionCudaCandidate::head_count(self : Self) -> Int {
  self.head_count
}

///|
pub fn GlmHybridKdaProjectionCudaCandidate::head_dimension(self : Self) -> Int {
  self.head_dimension
}

///|
pub fn GlmHybridKdaProjectionCudaCandidate::control_rank(self : Self) -> Int {
  self.control_rank
}

///|
pub fn GlmHybridKdaProjectionCudaCandidate::maximum_tokens(self : Self) -> Int {
  self.maximum_tokens
}

///|
pub fn GlmHybridKdaProjectionCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn GlmHybridKdaProjectionCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn GlmHybridKdaProjectionCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[@glm_hybrid_artifact_admission.GlmHybridLaunchOperand] {
  self.operands
}

///|
pub fn GlmHybridKdaProjectionCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn GlmHybridKdaProjectionCudaCandidate::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn GlmHybridKdaProjectionCudaCandidate::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn GlmHybridKdaProjectionCudaCandidate::source_digest(
  self : Self,
) -> GlmHybridCudaSourceDigest {
  self.source_digest
}

///|
pub fn GlmHybridKdaProjectionCudaCandidate::recipe_bytes(self : Self) -> Bytes {
  self.recipe
}

///|
pub fn GlmHybridKdaProjectionCudaCandidate::recipe_digest(
  self : Self,
) -> GlmHybridCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn GlmHybridKdaProjectionCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

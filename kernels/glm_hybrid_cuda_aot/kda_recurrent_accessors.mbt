///|
pub fn GlmHybridKdaRecurrentCudaCandidate::family(
  self : Self,
) -> GlmHybridKdaRecurrentAotFamily {
  self.family
}

///|
pub fn GlmHybridKdaRecurrentCudaCandidate::numerical_contract(
  self : Self,
) -> @recurrent_delta_cuda_source.RecurrentDeltaNumericalContract {
  self.numerical_contract
}

///|
pub fn GlmHybridKdaRecurrentCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn GlmHybridKdaRecurrentCudaCandidate::requirements_digest(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn GlmHybridKdaRecurrentCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn GlmHybridKdaRecurrentCudaCandidate::operation(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation {
  self.operation
}

///|
pub fn GlmHybridKdaRecurrentCudaCandidate::capability(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridAotCapability {
  self.capability
}

///|
pub fn GlmHybridKdaRecurrentCudaCandidate::head_count(self : Self) -> Int {
  self.head_count
}

///|
pub fn GlmHybridKdaRecurrentCudaCandidate::head_dimension(self : Self) -> Int {
  self.head_dimension
}

///|
pub fn GlmHybridKdaRecurrentCudaCandidate::normalization_epsilon(
  self : Self,
) -> Double {
  self.normalization_epsilon
}

///|
pub fn GlmHybridKdaRecurrentCudaCandidate::maximum_tokens(self : Self) -> Int {
  self.maximum_tokens
}

///|
pub fn GlmHybridKdaRecurrentCudaCandidate::maximum_sequences(
  self : Self,
) -> Int {
  self.maximum_sequences
}

///|
pub fn GlmHybridKdaRecurrentCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn GlmHybridKdaRecurrentCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn GlmHybridKdaRecurrentCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[@glm_hybrid_artifact_admission.GlmHybridLaunchOperand] {
  self.operands
}

///|
pub fn GlmHybridKdaRecurrentCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn GlmHybridKdaRecurrentCudaCandidate::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn GlmHybridKdaRecurrentCudaCandidate::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn GlmHybridKdaRecurrentCudaCandidate::source_digest(
  self : Self,
) -> GlmHybridCudaSourceDigest {
  self.source_digest
}

///|
pub fn GlmHybridKdaRecurrentCudaCandidate::recipe_bytes(self : Self) -> Bytes {
  self.recipe
}

///|
pub fn GlmHybridKdaRecurrentCudaCandidate::recipe_digest(
  self : Self,
) -> GlmHybridCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn GlmHybridKdaRecurrentCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

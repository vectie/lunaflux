///|
pub fn GlmHybridKdaGatedNormCudaCandidate::family(
  self : Self,
) -> GlmHybridKdaGatedNormAotFamily {
  self.family
}

///|
pub fn GlmHybridKdaGatedNormCudaCandidate::numerical_contract(
  self : Self,
) -> @gated_rms_norm_cuda_source.GatedRmsNormNumericalContract {
  self.numerical_contract
}

///|
pub fn GlmHybridKdaGatedNormCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn GlmHybridKdaGatedNormCudaCandidate::requirements_digest(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn GlmHybridKdaGatedNormCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn GlmHybridKdaGatedNormCudaCandidate::operation(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation {
  self.operation
}

///|
pub fn GlmHybridKdaGatedNormCudaCandidate::capability(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridAotCapability {
  self.capability
}

///|
pub fn GlmHybridKdaGatedNormCudaCandidate::head_count(self : Self) -> Int {
  self.head_count
}

///|
pub fn GlmHybridKdaGatedNormCudaCandidate::head_dimension(self : Self) -> Int {
  self.head_dimension
}

///|
pub fn GlmHybridKdaGatedNormCudaCandidate::epsilon(self : Self) -> Double {
  self.epsilon
}

///|
pub fn GlmHybridKdaGatedNormCudaCandidate::maximum_tokens(self : Self) -> Int {
  self.maximum_tokens
}

///|
pub fn GlmHybridKdaGatedNormCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn GlmHybridKdaGatedNormCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn GlmHybridKdaGatedNormCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[@glm_hybrid_artifact_admission.GlmHybridLaunchOperand] {
  self.operands
}

///|
pub fn GlmHybridKdaGatedNormCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn GlmHybridKdaGatedNormCudaCandidate::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn GlmHybridKdaGatedNormCudaCandidate::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn GlmHybridKdaGatedNormCudaCandidate::source_digest(
  self : Self,
) -> GlmHybridCudaSourceDigest {
  self.source_digest
}

///|
pub fn GlmHybridKdaGatedNormCudaCandidate::recipe_bytes(self : Self) -> Bytes {
  self.recipe
}

///|
pub fn GlmHybridKdaGatedNormCudaCandidate::recipe_digest(
  self : Self,
) -> GlmHybridCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn GlmHybridKdaGatedNormCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

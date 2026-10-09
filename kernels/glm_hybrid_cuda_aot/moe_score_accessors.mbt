///|
pub fn GlmHybridMoeScoreCudaCandidate::family(
  self : Self,
) -> GlmHybridMoeScoreAotFamily {
  self.family
}

///|
pub fn GlmHybridMoeScoreCudaCandidate::numerical_contract(
  self : Self,
) -> @sigmoid_correction_cuda_source.SigmoidCorrectionNumericalContract {
  self.numerical_contract
}

///|
pub fn GlmHybridMoeScoreCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn GlmHybridMoeScoreCudaCandidate::requirements_digest(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn GlmHybridMoeScoreCudaCandidate::requirement_ordinal(self : Self) -> Int {
  self.requirement_ordinal
}

///|
pub fn GlmHybridMoeScoreCudaCandidate::operation(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation {
  self.operation
}

///|
pub fn GlmHybridMoeScoreCudaCandidate::capability(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridAotCapability {
  self.capability
}

///|
pub fn GlmHybridMoeScoreCudaCandidate::expert_count(self : Self) -> Int {
  self.expert_count
}

///|
pub fn GlmHybridMoeScoreCudaCandidate::maximum_tokens(self : Self) -> Int {
  self.maximum_tokens
}

///|
pub fn GlmHybridMoeScoreCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn GlmHybridMoeScoreCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn GlmHybridMoeScoreCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[@glm_hybrid_artifact_admission.GlmHybridLaunchOperand] {
  self.operands
}

///|
pub fn GlmHybridMoeScoreCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn GlmHybridMoeScoreCudaCandidate::function_symbol(self : Self) -> String {
  self.function_symbol
}

///|
pub fn GlmHybridMoeScoreCudaCandidate::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn GlmHybridMoeScoreCudaCandidate::source_digest(
  self : Self,
) -> GlmHybridCudaSourceDigest {
  self.source_digest
}

///|
pub fn GlmHybridMoeScoreCudaCandidate::recipe_bytes(self : Self) -> Bytes {
  self.recipe
}

///|
pub fn GlmHybridMoeScoreCudaCandidate::recipe_digest(
  self : Self,
) -> GlmHybridCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn GlmHybridMoeScoreCudaCandidate::manifest_bindable(_self : Self) -> Bool {
  false
}

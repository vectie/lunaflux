///|
pub fn GlmHybridDenseSwiGluCudaCandidate::family(
  self : Self,
) -> GlmHybridDenseSwiGluAotFamily {
  self.family
}

///|
pub fn GlmHybridDenseSwiGluCudaCandidate::numerical_contract(
  self : Self,
) -> @gated_mlp_cuda_source.GatedMlpNumericalContract {
  self.numerical_contract
}

///|
pub fn GlmHybridDenseSwiGluCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn GlmHybridDenseSwiGluCudaCandidate::requirements_digest(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn GlmHybridDenseSwiGluCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn GlmHybridDenseSwiGluCudaCandidate::operation(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation {
  self.operation
}

///|
pub fn GlmHybridDenseSwiGluCudaCandidate::capability(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridAotCapability {
  self.capability
}

///|
pub fn GlmHybridDenseSwiGluCudaCandidate::hidden_size(self : Self) -> Int {
  self.hidden_size
}

///|
pub fn GlmHybridDenseSwiGluCudaCandidate::intermediate_size(self : Self) -> Int {
  self.intermediate_size
}

///|
pub fn GlmHybridDenseSwiGluCudaCandidate::maximum_tokens(self : Self) -> Int {
  self.maximum_tokens
}

///|
pub fn GlmHybridDenseSwiGluCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn GlmHybridDenseSwiGluCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn GlmHybridDenseSwiGluCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[@glm_hybrid_artifact_admission.GlmHybridLaunchOperand] {
  self.operands
}

///|
pub fn GlmHybridDenseSwiGluCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn GlmHybridDenseSwiGluCudaCandidate::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn GlmHybridDenseSwiGluCudaCandidate::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn GlmHybridDenseSwiGluCudaCandidate::source_digest(
  self : Self,
) -> GlmHybridCudaSourceDigest {
  self.source_digest
}

///|
pub fn GlmHybridDenseSwiGluCudaCandidate::recipe_bytes(self : Self) -> Bytes {
  self.recipe
}

///|
pub fn GlmHybridDenseSwiGluCudaCandidate::recipe_digest(
  self : Self,
) -> GlmHybridCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn GlmHybridDenseSwiGluCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

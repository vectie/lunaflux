///|
pub fn GlmHybridKdaOutputGateCudaCandidate::family(
  self : Self,
) -> GlmHybridKdaOutputGateAotFamily {
  self.family
}

///|
pub fn GlmHybridKdaOutputGateCudaCandidate::numerical_contract(
  self : Self,
) -> @two_stage_projection_cuda_source.TwoStageProjectionNumericalContract {
  self.numerical_contract
}

///|
pub fn GlmHybridKdaOutputGateCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn GlmHybridKdaOutputGateCudaCandidate::requirements_digest(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn GlmHybridKdaOutputGateCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn GlmHybridKdaOutputGateCudaCandidate::operation(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation {
  self.operation
}

///|
pub fn GlmHybridKdaOutputGateCudaCandidate::capability(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridAotCapability {
  self.capability
}

///|
pub fn GlmHybridKdaOutputGateCudaCandidate::hidden_width(self : Self) -> Int {
  self.hidden_width
}

///|
pub fn GlmHybridKdaOutputGateCudaCandidate::head_count(self : Self) -> Int {
  self.head_count
}

///|
pub fn GlmHybridKdaOutputGateCudaCandidate::head_dimension(self : Self) -> Int {
  self.head_dimension
}

///|
pub fn GlmHybridKdaOutputGateCudaCandidate::maximum_tokens(self : Self) -> Int {
  self.maximum_tokens
}

///|
pub fn GlmHybridKdaOutputGateCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn GlmHybridKdaOutputGateCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn GlmHybridKdaOutputGateCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[@glm_hybrid_artifact_admission.GlmHybridLaunchOperand] {
  self.operands
}

///|
pub fn GlmHybridKdaOutputGateCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn GlmHybridKdaOutputGateCudaCandidate::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn GlmHybridKdaOutputGateCudaCandidate::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn GlmHybridKdaOutputGateCudaCandidate::source_digest(
  self : Self,
) -> GlmHybridCudaSourceDigest {
  self.source_digest
}

///|
pub fn GlmHybridKdaOutputGateCudaCandidate::recipe_bytes(self : Self) -> Bytes {
  self.recipe
}

///|
pub fn GlmHybridKdaOutputGateCudaCandidate::recipe_digest(
  self : Self,
) -> GlmHybridCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn GlmHybridKdaOutputGateCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

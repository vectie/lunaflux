///|
pub fn GlmHybridMoeRouterCudaCandidate::family(
  self : Self,
) -> GlmHybridMoeRouterAotFamily {
  self.family
}

///|
pub fn GlmHybridMoeRouterCudaCandidate::numerical_contract(
  self : Self,
) -> @router_projection_cuda_source.RouterProjectionNumericalContract {
  self.numerical_contract
}

///|
pub fn GlmHybridMoeRouterCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn GlmHybridMoeRouterCudaCandidate::requirements_digest(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn GlmHybridMoeRouterCudaCandidate::requirement_ordinal(self : Self) -> Int {
  self.requirement_ordinal
}

///|
pub fn GlmHybridMoeRouterCudaCandidate::operation(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation {
  self.operation
}

///|
pub fn GlmHybridMoeRouterCudaCandidate::capability(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridAotCapability {
  self.capability
}

///|
pub fn GlmHybridMoeRouterCudaCandidate::hidden_width(self : Self) -> Int {
  self.hidden_width
}

///|
pub fn GlmHybridMoeRouterCudaCandidate::expert_count(self : Self) -> Int {
  self.expert_count
}

///|
pub fn GlmHybridMoeRouterCudaCandidate::maximum_tokens(self : Self) -> Int {
  self.maximum_tokens
}

///|
pub fn GlmHybridMoeRouterCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn GlmHybridMoeRouterCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn GlmHybridMoeRouterCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[@glm_hybrid_artifact_admission.GlmHybridLaunchOperand] {
  self.operands
}

///|
pub fn GlmHybridMoeRouterCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn GlmHybridMoeRouterCudaCandidate::function_symbol(self : Self) -> String {
  self.function_symbol
}

///|
pub fn GlmHybridMoeRouterCudaCandidate::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn GlmHybridMoeRouterCudaCandidate::source_digest(
  self : Self,
) -> GlmHybridCudaSourceDigest {
  self.source_digest
}

///|
pub fn GlmHybridMoeRouterCudaCandidate::recipe_bytes(self : Self) -> Bytes {
  self.recipe
}

///|
pub fn GlmHybridMoeRouterCudaCandidate::recipe_digest(
  self : Self,
) -> GlmHybridCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn GlmHybridMoeRouterCudaCandidate::manifest_bindable(_self : Self) -> Bool {
  false
}

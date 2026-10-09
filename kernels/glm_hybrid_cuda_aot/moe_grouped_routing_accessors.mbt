///|
pub fn GlmHybridMoeGroupedRoutingCudaCandidate::family(
  self : Self,
) -> GlmHybridMoeGroupedRoutingAotFamily {
  self.family
}

///|
pub fn GlmHybridMoeGroupedRoutingCudaCandidate::numerical_contract(
  self : Self,
) -> @grouped_routing_cuda_source.GroupedRoutingNumericalContract {
  self.numerical_contract
}

///|
pub fn GlmHybridMoeGroupedRoutingCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn GlmHybridMoeGroupedRoutingCudaCandidate::requirements_digest(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn GlmHybridMoeGroupedRoutingCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn GlmHybridMoeGroupedRoutingCudaCandidate::operation(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation {
  self.operation
}

///|
pub fn GlmHybridMoeGroupedRoutingCudaCandidate::capability(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridAotCapability {
  self.capability
}

///|
pub fn GlmHybridMoeGroupedRoutingCudaCandidate::routed_expert_count(
  self : Self,
) -> Int {
  self.routed_expert_count
}

///|
pub fn GlmHybridMoeGroupedRoutingCudaCandidate::maximum_tokens(
  self : Self,
) -> Int {
  self.maximum_tokens
}

///|
pub fn GlmHybridMoeGroupedRoutingCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn GlmHybridMoeGroupedRoutingCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn GlmHybridMoeGroupedRoutingCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[@glm_hybrid_artifact_admission.GlmHybridLaunchOperand] {
  self.operands
}

///|
pub fn GlmHybridMoeGroupedRoutingCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn GlmHybridMoeGroupedRoutingCudaCandidate::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn GlmHybridMoeGroupedRoutingCudaCandidate::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

///|
pub fn GlmHybridMoeGroupedRoutingCudaCandidate::source_digest(
  self : Self,
) -> GlmHybridCudaSourceDigest {
  self.source_digest
}

///|
pub fn GlmHybridMoeGroupedRoutingCudaCandidate::recipe_bytes(
  self : Self,
) -> Bytes {
  self.recipe
}

///|
pub fn GlmHybridMoeGroupedRoutingCudaCandidate::recipe_digest(
  self : Self,
) -> GlmHybridCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn GlmHybridMoeGroupedRoutingCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

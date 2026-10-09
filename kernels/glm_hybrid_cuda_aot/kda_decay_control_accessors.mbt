///|
pub fn GlmHybridKdaDecayControlCudaCandidate::family(
  self : Self,
) -> GlmHybridKdaDecayControlAotFamily {
  self.family
}

///|
pub fn GlmHybridKdaDecayControlCudaCandidate::numerical_contract(
  self : Self,
) -> @decay_control_cuda_source.DecayControlNumericalContract {
  self.numerical_contract
}

///|
pub fn GlmHybridKdaDecayControlCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn GlmHybridKdaDecayControlCudaCandidate::requirements_digest(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn GlmHybridKdaDecayControlCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn GlmHybridKdaDecayControlCudaCandidate::operation(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation {
  self.operation
}

///|
pub fn GlmHybridKdaDecayControlCudaCandidate::capability(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridAotCapability {
  self.capability
}

///|
pub fn GlmHybridKdaDecayControlCudaCandidate::head_count(self : Self) -> Int {
  self.head_count
}

///|
pub fn GlmHybridKdaDecayControlCudaCandidate::head_dimension(
  self : Self,
) -> Int {
  self.head_dimension
}

///|
pub fn GlmHybridKdaDecayControlCudaCandidate::lower_bound(
  self : Self,
) -> Double {
  self.lower_bound
}

///|
pub fn GlmHybridKdaDecayControlCudaCandidate::maximum_tokens(
  self : Self,
) -> Int {
  self.maximum_tokens
}

///|
pub fn GlmHybridKdaDecayControlCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn GlmHybridKdaDecayControlCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn GlmHybridKdaDecayControlCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[@glm_hybrid_artifact_admission.GlmHybridLaunchOperand] {
  self.operands
}

///|
pub fn GlmHybridKdaDecayControlCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn GlmHybridKdaDecayControlCudaCandidate::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn GlmHybridKdaDecayControlCudaCandidate::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

///|
pub fn GlmHybridKdaDecayControlCudaCandidate::source_digest(
  self : Self,
) -> GlmHybridCudaSourceDigest {
  self.source_digest
}

///|
pub fn GlmHybridKdaDecayControlCudaCandidate::recipe_bytes(
  self : Self,
) -> Bytes {
  self.recipe
}

///|
pub fn GlmHybridKdaDecayControlCudaCandidate::recipe_digest(
  self : Self,
) -> GlmHybridCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn GlmHybridKdaDecayControlCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

///|
pub fn GlmHybridHyperControlCudaCandidate::family(
  self : Self,
) -> GlmHybridHyperControlAotFamily {
  self.family
}

///|
pub fn GlmHybridHyperControlCudaCandidate::numerical_contract(
  self : Self,
) -> @decoder_hyper_connection_cuda_source.DecoderHyperConnectionNumericalContract {
  self.numerical_contract
}

///|
pub fn GlmHybridHyperControlCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn GlmHybridHyperControlCudaCandidate::requirements_digest(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn GlmHybridHyperControlCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn GlmHybridHyperControlCudaCandidate::operation(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation {
  self.operation
}

///|
pub fn GlmHybridHyperControlCudaCandidate::capability(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridAotCapability {
  self.capability
}

///|
pub fn GlmHybridHyperControlCudaCandidate::stream_count(self : Self) -> Int {
  self.stream_count
}

///|
pub fn GlmHybridHyperControlCudaCandidate::hidden_width(self : Self) -> Int {
  self.hidden_width
}

///|
pub fn GlmHybridHyperControlCudaCandidate::sinkhorn_iterations(
  self : Self,
) -> Int {
  self.sinkhorn_iterations
}

///|
pub fn GlmHybridHyperControlCudaCandidate::control_epsilon(
  self : Self,
) -> Double {
  self.control_epsilon
}

///|
pub fn GlmHybridHyperControlCudaCandidate::normalization_epsilon(
  self : Self,
) -> Double {
  self.normalization_epsilon
}

///|
pub fn GlmHybridHyperControlCudaCandidate::maximum_tokens(self : Self) -> Int {
  self.maximum_tokens
}

///|
pub fn GlmHybridHyperControlCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn GlmHybridHyperControlCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn GlmHybridHyperControlCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[@glm_hybrid_artifact_admission.GlmHybridLaunchOperand] {
  self.operands
}

///|
pub fn GlmHybridHyperControlCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn GlmHybridHyperControlCudaCandidate::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn GlmHybridHyperControlCudaCandidate::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn GlmHybridHyperControlCudaCandidate::source_digest(
  self : Self,
) -> GlmHybridCudaSourceDigest {
  self.source_digest
}

///|
pub fn GlmHybridHyperControlCudaCandidate::recipe_bytes(self : Self) -> Bytes {
  self.recipe
}

///|
pub fn GlmHybridHyperControlCudaCandidate::recipe_digest(
  self : Self,
) -> GlmHybridCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn GlmHybridHyperControlCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

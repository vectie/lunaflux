///|
pub fn GlmHybridKdaShortConvolutionCudaCandidate::family(
  self : Self,
) -> GlmHybridKdaShortConvolutionAotFamily {
  self.family
}

///|
pub fn GlmHybridKdaShortConvolutionCudaCandidate::numerical_contract(
  self : Self,
) -> @causal_short_convolution_cuda_source.CausalShortConvolutionNumericalContract {
  self.numerical_contract
}

///|
pub fn GlmHybridKdaShortConvolutionCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn GlmHybridKdaShortConvolutionCudaCandidate::requirements_digest(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn GlmHybridKdaShortConvolutionCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn GlmHybridKdaShortConvolutionCudaCandidate::operation(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation {
  self.operation
}

///|
pub fn GlmHybridKdaShortConvolutionCudaCandidate::capability(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridAotCapability {
  self.capability
}

///|
pub fn GlmHybridKdaShortConvolutionCudaCandidate::channel_count(
  self : Self,
) -> Int {
  self.channel_count
}

///|
pub fn GlmHybridKdaShortConvolutionCudaCandidate::kernel_size(
  self : Self,
) -> Int {
  self.kernel_size
}

///|
pub fn GlmHybridKdaShortConvolutionCudaCandidate::maximum_tokens(
  self : Self,
) -> Int {
  self.maximum_tokens
}

///|
pub fn GlmHybridKdaShortConvolutionCudaCandidate::maximum_sequences(
  self : Self,
) -> Int {
  self.maximum_sequences
}

///|
pub fn GlmHybridKdaShortConvolutionCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn GlmHybridKdaShortConvolutionCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn GlmHybridKdaShortConvolutionCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[@glm_hybrid_artifact_admission.GlmHybridLaunchOperand] {
  self.operands
}

///|
pub fn GlmHybridKdaShortConvolutionCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn GlmHybridKdaShortConvolutionCudaCandidate::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn GlmHybridKdaShortConvolutionCudaCandidate::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

///|
pub fn GlmHybridKdaShortConvolutionCudaCandidate::source_digest(
  self : Self,
) -> GlmHybridCudaSourceDigest {
  self.source_digest
}

///|
pub fn GlmHybridKdaShortConvolutionCudaCandidate::recipe_bytes(
  self : Self,
) -> Bytes {
  self.recipe
}

///|
pub fn GlmHybridKdaShortConvolutionCudaCandidate::recipe_digest(
  self : Self,
) -> GlmHybridCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn GlmHybridKdaShortConvolutionCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

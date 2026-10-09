///|
pub fn GlmHybridDenseProjectionCudaCandidate::family(
  self : Self,
) -> GlmHybridDenseProjectionAotFamily {
  self.family
}

///|
pub fn GlmHybridDenseProjectionCudaCandidate::numerical_contract(
  self : Self,
) -> @decoder_projection_cuda_source.DecoderProjectionNumericalContract {
  self.numerical_contract
}

///|
pub fn GlmHybridDenseProjectionCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn GlmHybridDenseProjectionCudaCandidate::requirements_digest(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn GlmHybridDenseProjectionCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn GlmHybridDenseProjectionCudaCandidate::operation(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation {
  self.operation
}

///|
pub fn GlmHybridDenseProjectionCudaCandidate::capability(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridAotCapability {
  self.capability
}

///|
pub fn GlmHybridDenseProjectionCudaCandidate::input_width(self : Self) -> Int {
  self.input_width
}

///|
pub fn GlmHybridDenseProjectionCudaCandidate::output_width(self : Self) -> Int {
  self.output_width
}

///|
pub fn GlmHybridDenseProjectionCudaCandidate::maximum_tokens(
  self : Self,
) -> Int {
  self.maximum_tokens
}

///|
pub fn GlmHybridDenseProjectionCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn GlmHybridDenseProjectionCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn GlmHybridDenseProjectionCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[@glm_hybrid_artifact_admission.GlmHybridLaunchOperand] {
  self.operands
}

///|
pub fn GlmHybridDenseProjectionCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn GlmHybridDenseProjectionCudaCandidate::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn GlmHybridDenseProjectionCudaCandidate::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

///|
pub fn GlmHybridDenseProjectionCudaCandidate::source_digest(
  self : Self,
) -> GlmHybridCudaSourceDigest {
  self.source_digest
}

///|
pub fn GlmHybridDenseProjectionCudaCandidate::recipe_bytes(
  self : Self,
) -> Bytes {
  self.recipe
}

///|
pub fn GlmHybridDenseProjectionCudaCandidate::recipe_digest(
  self : Self,
) -> GlmHybridCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn GlmHybridDenseProjectionCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

///|
pub fn GlmHybridRotaryCudaCandidate::family(
  self : Self,
) -> GlmHybridRotaryAotFamily {
  self.family
}

///|
pub fn GlmHybridRotaryCudaCandidate::numerical_contract(
  self : Self,
) -> @decoder_interleaved_rope_cuda_source.DecoderInterleavedRopeNumericalContract {
  self.numerical_contract
}

///|
pub fn GlmHybridRotaryCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn GlmHybridRotaryCudaCandidate::requirements_digest(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn GlmHybridRotaryCudaCandidate::requirement_ordinal(self : Self) -> Int {
  self.requirement_ordinal
}

///|
pub fn GlmHybridRotaryCudaCandidate::operation(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation {
  self.operation
}

///|
pub fn GlmHybridRotaryCudaCandidate::capability(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridAotCapability {
  self.capability
}

///|
pub fn GlmHybridRotaryCudaCandidate::query_head_count(self : Self) -> Int {
  self.query_head_count
}

///|
pub fn GlmHybridRotaryCudaCandidate::query_head_dimension(self : Self) -> Int {
  self.query_head_dimension
}

///|
pub fn GlmHybridRotaryCudaCandidate::rope_dimension(self : Self) -> Int {
  self.rope_dimension
}

///|
pub fn GlmHybridRotaryCudaCandidate::maximum_tokens(self : Self) -> Int {
  self.maximum_tokens
}

///|
pub fn GlmHybridRotaryCudaCandidate::maximum_positions(self : Self) -> Int {
  self.maximum_positions
}

///|
pub fn GlmHybridRotaryCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn GlmHybridRotaryCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn GlmHybridRotaryCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[@glm_hybrid_artifact_admission.GlmHybridLaunchOperand] {
  self.operands
}

///|
pub fn GlmHybridRotaryCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn GlmHybridRotaryCudaCandidate::function_symbol(self : Self) -> String {
  self.function_symbol
}

///|
pub fn GlmHybridRotaryCudaCandidate::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn GlmHybridRotaryCudaCandidate::source_digest(
  self : Self,
) -> GlmHybridCudaSourceDigest {
  self.source_digest
}

///|
pub fn GlmHybridRotaryCudaCandidate::recipe_bytes(self : Self) -> Bytes {
  self.recipe
}

///|
pub fn GlmHybridRotaryCudaCandidate::recipe_digest(
  self : Self,
) -> GlmHybridCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn GlmHybridRotaryCudaCandidate::manifest_bindable(_self : Self) -> Bool {
  false
}

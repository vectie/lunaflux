///|
pub fn GlmHybridKvSplitNormCudaCandidate::family(
  self : Self,
) -> GlmHybridKvSplitNormAotFamily {
  self.family
}

///|
pub fn GlmHybridKvSplitNormCudaCandidate::numerical_contract(
  self : Self,
) -> @prefix_rms_norm_cuda_source.PrefixRmsNormNumericalContract {
  self.numerical_contract
}

///|
pub fn GlmHybridKvSplitNormCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn GlmHybridKvSplitNormCudaCandidate::requirements_digest(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn GlmHybridKvSplitNormCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn GlmHybridKvSplitNormCudaCandidate::operation(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation {
  self.operation
}

///|
pub fn GlmHybridKvSplitNormCudaCandidate::capability(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridAotCapability {
  self.capability
}

///|
pub fn GlmHybridKvSplitNormCudaCandidate::prefix_width(self : Self) -> Int {
  self.prefix_width
}

///|
pub fn GlmHybridKvSplitNormCudaCandidate::suffix_width(self : Self) -> Int {
  self.suffix_width
}

///|
pub fn GlmHybridKvSplitNormCudaCandidate::maximum_tokens(self : Self) -> Int {
  self.maximum_tokens
}

///|
pub fn GlmHybridKvSplitNormCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn GlmHybridKvSplitNormCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn GlmHybridKvSplitNormCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[@glm_hybrid_artifact_admission.GlmHybridLaunchOperand] {
  self.operands
}

///|
pub fn GlmHybridKvSplitNormCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn GlmHybridKvSplitNormCudaCandidate::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn GlmHybridKvSplitNormCudaCandidate::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn GlmHybridKvSplitNormCudaCandidate::source_digest(
  self : Self,
) -> GlmHybridCudaSourceDigest {
  self.source_digest
}

///|
pub fn GlmHybridKvSplitNormCudaCandidate::recipe_bytes(self : Self) -> Bytes {
  self.recipe
}

///|
pub fn GlmHybridKvSplitNormCudaCandidate::recipe_digest(
  self : Self,
) -> GlmHybridCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn GlmHybridKvSplitNormCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

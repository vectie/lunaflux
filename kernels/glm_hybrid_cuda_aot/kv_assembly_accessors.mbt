///|
pub fn GlmHybridKvAssemblyCudaCandidate::family(
  self : Self,
) -> GlmHybridKvAssemblyAotFamily {
  self.family
}

///|
pub fn GlmHybridKvAssemblyCudaCandidate::numerical_contract(
  self : Self,
) -> @decoder_kv_assembly_cuda_source.DecoderKvAssemblyNumericalContract {
  self.numerical_contract
}

///|
pub fn GlmHybridKvAssemblyCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn GlmHybridKvAssemblyCudaCandidate::requirements_digest(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn GlmHybridKvAssemblyCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn GlmHybridKvAssemblyCudaCandidate::operation(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation {
  self.operation
}

///|
pub fn GlmHybridKvAssemblyCudaCandidate::capability(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridAotCapability {
  self.capability
}

///|
pub fn GlmHybridKvAssemblyCudaCandidate::maximum_tokens(self : Self) -> Int {
  self.maximum_tokens
}

///|
pub fn GlmHybridKvAssemblyCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn GlmHybridKvAssemblyCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn GlmHybridKvAssemblyCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[@glm_hybrid_artifact_admission.GlmHybridLaunchOperand] {
  self.operands
}

///|
pub fn GlmHybridKvAssemblyCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn GlmHybridKvAssemblyCudaCandidate::function_symbol(self : Self) -> String {
  self.function_symbol
}

///|
pub fn GlmHybridKvAssemblyCudaCandidate::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn GlmHybridKvAssemblyCudaCandidate::source_digest(
  self : Self,
) -> GlmHybridCudaSourceDigest {
  self.source_digest
}

///|
pub fn GlmHybridKvAssemblyCudaCandidate::recipe_bytes(self : Self) -> Bytes {
  self.recipe
}

///|
pub fn GlmHybridKvAssemblyCudaCandidate::recipe_digest(
  self : Self,
) -> GlmHybridCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn GlmHybridKvAssemblyCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

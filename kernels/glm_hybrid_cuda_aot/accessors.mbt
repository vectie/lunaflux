///|
pub fn GlmHybridCudaAotLimits::new(
  maximum_tokens~ : Int,
  maximum_source_bytes~ : Int,
  maximum_recipe_bytes~ : Int,
) -> Self raise GlmHybridCudaAotError {
  if maximum_tokens <= 0 ||
    maximum_tokens > 1048576 ||
    maximum_source_bytes <= 0 ||
    maximum_recipe_bytes <= 0 {
    raise Invalid(Limit)
  }
  { maximum_tokens, maximum_source_bytes, maximum_recipe_bytes, }
}

///|
pub fn GlmHybridFoundationOperand::role(
  self : Self,
) -> GlmHybridFoundationOperandRole {
  self.role
}

///|
pub fn GlmHybridFoundationOperand::byte_count(self : Self) -> Int64 {
  self.byte_count
}

///|
pub fn GlmHybridFoundationOperand::alignment(self : Self) -> Int64 {
  self.alignment
}

///|
pub fn GlmHybridFoundationCudaCandidate::family(
  self : Self,
) -> GlmHybridFoundationAotFamily {
  self.family
}

///|
pub fn GlmHybridFoundationCudaCandidate::numerical_contract(
  self : Self,
) -> @decoder_foundation_cuda_source.DecoderFoundationNumericalContract {
  self.numerical_contract
}

///|
pub fn GlmHybridFoundationCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn GlmHybridFoundationCudaCandidate::requirements_digest(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn GlmHybridFoundationCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn GlmHybridFoundationCudaCandidate::operation(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation {
  self.operation
}

///|
pub fn GlmHybridFoundationCudaCandidate::capability(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridAotCapability {
  self.capability
}

///|
pub fn GlmHybridFoundationCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn GlmHybridFoundationCudaCandidate::maximum_tokens(self : Self) -> Int {
  self.maximum_tokens
}

///|
pub fn GlmHybridFoundationCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn GlmHybridFoundationCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[GlmHybridFoundationOperand] {
  self.operands
}

///|
pub fn GlmHybridFoundationCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn GlmHybridFoundationCudaCandidate::function_symbol(self : Self) -> String {
  self.function_symbol
}

///|
pub fn GlmHybridFoundationCudaCandidate::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn GlmHybridFoundationCudaCandidate::source_digest(
  self : Self,
) -> GlmHybridCudaSourceDigest {
  self.source_digest
}

///|
pub fn GlmHybridFoundationCudaCandidate::recipe_bytes(self : Self) -> Bytes {
  self.recipe
}

///|
pub fn GlmHybridFoundationCudaCandidate::recipe_digest(
  self : Self,
) -> GlmHybridCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn GlmHybridFoundationCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

///|
pub fn GlmHybridCudaSourceDigest::as_hex(self : Self) -> String {
  self.0
}

///|
pub fn GlmHybridCudaRecipeDigest::as_hex(self : Self) -> String {
  self.0
}

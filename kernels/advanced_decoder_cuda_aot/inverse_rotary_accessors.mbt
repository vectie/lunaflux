///|
pub fn AdvancedDecoderInverseRotaryCudaCandidate::family(
  self : Self,
) -> AdvancedDecoderInverseRotaryAotFamily {
  self.family
}

///|
pub fn AdvancedDecoderInverseRotaryCudaCandidate::numerical_contract(
  self : Self,
) -> AdvancedDecoderInverseRotaryNumericalContract {
  self.numerical_contract
}

///|
pub fn AdvancedDecoderInverseRotaryCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn AdvancedDecoderInverseRotaryCudaCandidate::requirements_digest(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn AdvancedDecoderInverseRotaryCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn AdvancedDecoderInverseRotaryCudaCandidate::operation(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation {
  self.operation
}

///|
pub fn AdvancedDecoderInverseRotaryCudaCandidate::capability(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderAotCapability {
  self.capability
}

///|
pub fn AdvancedDecoderInverseRotaryCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn AdvancedDecoderInverseRotaryCudaCandidate::maximum_tokens(
  self : Self,
) -> Int {
  self.maximum_tokens
}

///|
pub fn AdvancedDecoderInverseRotaryCudaCandidate::head_count(
  self : Self,
) -> Int {
  self.head_count
}

///|
pub fn AdvancedDecoderInverseRotaryCudaCandidate::head_dimension(
  self : Self,
) -> Int {
  self.head_dimension
}

///|
pub fn AdvancedDecoderInverseRotaryCudaCandidate::rope_dimension(
  self : Self,
) -> Int {
  self.rope_dimension
}

///|
pub fn AdvancedDecoderInverseRotaryCudaCandidate::maximum_positions(
  self : Self,
) -> Int {
  self.maximum_positions
}

///|
pub fn AdvancedDecoderInverseRotaryCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn AdvancedDecoderInverseRotaryCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[
  @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
] {
  self.operands
}

///|
pub fn AdvancedDecoderInverseRotaryCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn AdvancedDecoderInverseRotaryCudaCandidate::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn AdvancedDecoderInverseRotaryCudaCandidate::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

///|
pub fn AdvancedDecoderInverseRotaryCudaCandidate::source_digest(
  self : Self,
) -> AdvancedDecoderCudaSourceDigest {
  self.source_digest
}

///|
pub fn AdvancedDecoderInverseRotaryCudaCandidate::recipe_bytes(
  self : Self,
) -> Bytes {
  self.recipe
}

///|
pub fn AdvancedDecoderInverseRotaryCudaCandidate::recipe_digest(
  self : Self,
) -> AdvancedDecoderCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn AdvancedDecoderInverseRotaryCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

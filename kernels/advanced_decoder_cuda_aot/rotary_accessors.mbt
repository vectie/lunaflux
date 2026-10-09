///|
pub fn AdvancedDecoderRotaryCudaCandidate::family(
  self : Self,
) -> AdvancedDecoderRotaryAotFamily {
  self.family
}

///|
pub fn AdvancedDecoderRotaryCudaCandidate::numerical_contract(
  self : Self,
) -> AdvancedDecoderRotaryNumericalContract {
  self.numerical_contract
}

///|
pub fn AdvancedDecoderRotaryCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn AdvancedDecoderRotaryCudaCandidate::requirements_digest(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn AdvancedDecoderRotaryCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn AdvancedDecoderRotaryCudaCandidate::operation(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation {
  self.operation
}

///|
pub fn AdvancedDecoderRotaryCudaCandidate::capability(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderAotCapability {
  self.capability
}

///|
pub fn AdvancedDecoderRotaryCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn AdvancedDecoderRotaryCudaCandidate::maximum_tokens(self : Self) -> Int {
  self.maximum_tokens
}

///|
pub fn AdvancedDecoderRotaryCudaCandidate::query_head_count(self : Self) -> Int {
  self.query_head_count
}

///|
pub fn AdvancedDecoderRotaryCudaCandidate::key_value_head_count(
  self : Self,
) -> Int {
  self.key_value_head_count
}

///|
pub fn AdvancedDecoderRotaryCudaCandidate::rope_dimension(self : Self) -> Int {
  self.rope_dimension
}

///|
pub fn AdvancedDecoderRotaryCudaCandidate::maximum_positions(
  self : Self,
) -> Int {
  self.maximum_positions
}

///|
pub fn AdvancedDecoderRotaryCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn AdvancedDecoderRotaryCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[
  @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
] {
  self.operands
}

///|
pub fn AdvancedDecoderRotaryCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn AdvancedDecoderRotaryCudaCandidate::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn AdvancedDecoderRotaryCudaCandidate::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn AdvancedDecoderRotaryCudaCandidate::source_digest(
  self : Self,
) -> AdvancedDecoderCudaSourceDigest {
  self.source_digest
}

///|
pub fn AdvancedDecoderRotaryCudaCandidate::recipe_bytes(self : Self) -> Bytes {
  self.recipe
}

///|
pub fn AdvancedDecoderRotaryCudaCandidate::recipe_digest(
  self : Self,
) -> AdvancedDecoderCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn AdvancedDecoderRotaryCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

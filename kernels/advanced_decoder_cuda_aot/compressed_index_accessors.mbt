///|
pub fn AdvancedDecoderCompressedIndexCudaCandidate::family(
  self : Self,
) -> AdvancedDecoderCompressedIndexAotFamily {
  self.family
}

///|
pub fn AdvancedDecoderCompressedIndexCudaCandidate::numerical_contract(
  self : Self,
) -> AdvancedDecoderCompressedIndexNumericalContract {
  self.numerical_contract
}

///|
pub fn AdvancedDecoderCompressedIndexCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn AdvancedDecoderCompressedIndexCudaCandidate::requirements_digest(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn AdvancedDecoderCompressedIndexCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn AdvancedDecoderCompressedIndexCudaCandidate::operation(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation {
  self.operation
}

///|
pub fn AdvancedDecoderCompressedIndexCudaCandidate::capability(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderAotCapability {
  self.capability
}

///|
pub fn AdvancedDecoderCompressedIndexCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn AdvancedDecoderCompressedIndexCudaCandidate::maximum_tokens(
  self : Self,
) -> Int {
  self.maximum_tokens
}

///|
pub fn AdvancedDecoderCompressedIndexCudaCandidate::compression_ratio(
  self : Self,
) -> Int {
  self.compression_ratio
}

///|
pub fn AdvancedDecoderCompressedIndexCudaCandidate::maximum_position_count(
  self : Self,
) -> Int {
  self.maximum_position_count
}

///|
pub fn AdvancedDecoderCompressedIndexCudaCandidate::maximum_slots_per_row(
  self : Self,
) -> Int {
  self.maximum_slots_per_row
}

///|
pub fn AdvancedDecoderCompressedIndexCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn AdvancedDecoderCompressedIndexCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[
  @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
] {
  self.operands
}

///|
pub fn AdvancedDecoderCompressedIndexCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn AdvancedDecoderCompressedIndexCudaCandidate::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn AdvancedDecoderCompressedIndexCudaCandidate::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

///|
pub fn AdvancedDecoderCompressedIndexCudaCandidate::source_digest(
  self : Self,
) -> AdvancedDecoderCudaSourceDigest {
  self.source_digest
}

///|
pub fn AdvancedDecoderCompressedIndexCudaCandidate::recipe_bytes(
  self : Self,
) -> Bytes {
  self.recipe
}

///|
pub fn AdvancedDecoderCompressedIndexCudaCandidate::recipe_digest(
  self : Self,
) -> AdvancedDecoderCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn AdvancedDecoderCompressedIndexCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

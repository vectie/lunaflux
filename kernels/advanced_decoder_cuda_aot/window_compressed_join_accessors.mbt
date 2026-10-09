///|
pub fn AdvancedDecoderWindowCompressedJoinCudaCandidate::family(
  self : Self,
) -> AdvancedDecoderWindowCompressedJoinAotFamily {
  self.family
}

///|
pub fn AdvancedDecoderWindowCompressedJoinCudaCandidate::numerical_contract(
  self : Self,
) -> AdvancedDecoderWindowCompressedJoinNumericalContract {
  self.numerical_contract
}

///|
pub fn AdvancedDecoderWindowCompressedJoinCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn AdvancedDecoderWindowCompressedJoinCudaCandidate::requirements_digest(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn AdvancedDecoderWindowCompressedJoinCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn AdvancedDecoderWindowCompressedJoinCudaCandidate::operation(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation {
  self.operation
}

///|
pub fn AdvancedDecoderWindowCompressedJoinCudaCandidate::capability(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderAotCapability {
  self.capability
}

///|
pub fn AdvancedDecoderWindowCompressedJoinCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn AdvancedDecoderWindowCompressedJoinCudaCandidate::maximum_tokens(
  self : Self,
) -> Int {
  self.maximum_tokens
}

///|
pub fn AdvancedDecoderWindowCompressedJoinCudaCandidate::window_size(
  self : Self,
) -> Int {
  self.window_size
}

///|
pub fn AdvancedDecoderWindowCompressedJoinCudaCandidate::compression_ratio(
  self : Self,
) -> Int {
  self.compression_ratio
}

///|
pub fn AdvancedDecoderWindowCompressedJoinCudaCandidate::maximum_position_count(
  self : Self,
) -> Int {
  self.maximum_position_count
}

///|
pub fn AdvancedDecoderWindowCompressedJoinCudaCandidate::maximum_compressed_width(
  self : Self,
) -> Int {
  self.maximum_compressed_width
}

///|
pub fn AdvancedDecoderWindowCompressedJoinCudaCandidate::maximum_joined_width(
  self : Self,
) -> Int {
  self.maximum_joined_width
}

///|
pub fn AdvancedDecoderWindowCompressedJoinCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn AdvancedDecoderWindowCompressedJoinCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[
  @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
] {
  self.operands
}

///|
pub fn AdvancedDecoderWindowCompressedJoinCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn AdvancedDecoderWindowCompressedJoinCudaCandidate::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn AdvancedDecoderWindowCompressedJoinCudaCandidate::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

///|
pub fn AdvancedDecoderWindowCompressedJoinCudaCandidate::source_digest(
  self : Self,
) -> AdvancedDecoderCudaSourceDigest {
  self.source_digest
}

///|
pub fn AdvancedDecoderWindowCompressedJoinCudaCandidate::recipe_bytes(
  self : Self,
) -> Bytes {
  self.recipe
}

///|
pub fn AdvancedDecoderWindowCompressedJoinCudaCandidate::recipe_digest(
  self : Self,
) -> AdvancedDecoderCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn AdvancedDecoderWindowCompressedJoinCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

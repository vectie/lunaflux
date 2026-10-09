///|
pub fn AdvancedDecoderKeyValueProjectionCudaCandidate::family(
  self : Self,
) -> AdvancedDecoderKeyValueProjectionFamily {
  self.family
}

///|
pub fn AdvancedDecoderKeyValueProjectionCudaCandidate::numerical_contract(
  self : Self,
) -> AdvancedDecoderKeyValueProjectionNumericalContract {
  self.numerical_contract
}

///|
pub fn AdvancedDecoderKeyValueProjectionCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn AdvancedDecoderKeyValueProjectionCudaCandidate::requirements_digest(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn AdvancedDecoderKeyValueProjectionCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn AdvancedDecoderKeyValueProjectionCudaCandidate::operation(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation {
  self.operation
}

///|
pub fn AdvancedDecoderKeyValueProjectionCudaCandidate::capability(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderAotCapability {
  self.capability
}

///|
pub fn AdvancedDecoderKeyValueProjectionCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn AdvancedDecoderKeyValueProjectionCudaCandidate::maximum_tokens(
  self : Self,
) -> Int {
  self.maximum_tokens
}

///|
pub fn AdvancedDecoderKeyValueProjectionCudaCandidate::input_width(
  self : Self,
) -> Int {
  self.input_width
}

///|
pub fn AdvancedDecoderKeyValueProjectionCudaCandidate::output_width(
  self : Self,
) -> Int {
  self.output_width
}

///|
pub fn AdvancedDecoderKeyValueProjectionCudaCandidate::rope_width(
  self : Self,
) -> Int {
  self.rope_width
}

///|
pub fn AdvancedDecoderKeyValueProjectionCudaCandidate::block_width(
  self : Self,
) -> Int {
  self.block_width
}

///|
pub fn AdvancedDecoderKeyValueProjectionCudaCandidate::tensor_parallel_replicated(
  _self : Self,
) -> Bool {
  true
}

///|
pub fn AdvancedDecoderKeyValueProjectionCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn AdvancedDecoderKeyValueProjectionCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[
  @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
] {
  self.operands
}

///|
pub fn AdvancedDecoderKeyValueProjectionCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn AdvancedDecoderKeyValueProjectionCudaCandidate::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn AdvancedDecoderKeyValueProjectionCudaCandidate::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

///|
pub fn AdvancedDecoderKeyValueProjectionCudaCandidate::source_digest(
  self : Self,
) -> AdvancedDecoderCudaSourceDigest {
  self.source_digest
}

///|
pub fn AdvancedDecoderKeyValueProjectionCudaCandidate::recipe_bytes(
  self : Self,
) -> Bytes {
  self.recipe
}

///|
pub fn AdvancedDecoderKeyValueProjectionCudaCandidate::recipe_digest(
  self : Self,
) -> AdvancedDecoderCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn AdvancedDecoderKeyValueProjectionCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

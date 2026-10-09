///|
pub fn AdvancedDecoderOutputAProjectionCudaCandidate::family(
  self : Self,
) -> AdvancedDecoderOutputAProjectionFamily {
  self.family
}

///|
pub fn AdvancedDecoderOutputAProjectionCudaCandidate::numerical_contract(
  self : Self,
) -> AdvancedDecoderOutputAProjectionNumericalContract {
  self.numerical_contract
}

///|
pub fn AdvancedDecoderOutputAProjectionCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn AdvancedDecoderOutputAProjectionCudaCandidate::requirements_digest(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn AdvancedDecoderOutputAProjectionCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn AdvancedDecoderOutputAProjectionCudaCandidate::operation(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation {
  self.operation
}

///|
pub fn AdvancedDecoderOutputAProjectionCudaCandidate::capability(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderAotCapability {
  self.capability
}

///|
pub fn AdvancedDecoderOutputAProjectionCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn AdvancedDecoderOutputAProjectionCudaCandidate::maximum_tokens(
  self : Self,
) -> Int {
  self.maximum_tokens
}

///|
pub fn AdvancedDecoderOutputAProjectionCudaCandidate::group_count(
  self : Self,
) -> Int {
  self.group_count
}

///|
pub fn AdvancedDecoderOutputAProjectionCudaCandidate::input_width_per_group(
  self : Self,
) -> Int {
  self.input_width_per_group
}

///|
pub fn AdvancedDecoderOutputAProjectionCudaCandidate::output_rank(
  self : Self,
) -> Int {
  self.output_rank
}

///|
pub fn AdvancedDecoderOutputAProjectionCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn AdvancedDecoderOutputAProjectionCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[
  @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
] {
  self.operands
}

///|
pub fn AdvancedDecoderOutputAProjectionCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn AdvancedDecoderOutputAProjectionCudaCandidate::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn AdvancedDecoderOutputAProjectionCudaCandidate::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

///|
pub fn AdvancedDecoderOutputAProjectionCudaCandidate::source_digest(
  self : Self,
) -> AdvancedDecoderCudaSourceDigest {
  self.source_digest
}

///|
pub fn AdvancedDecoderOutputAProjectionCudaCandidate::recipe_bytes(
  self : Self,
) -> Bytes {
  self.recipe
}

///|
pub fn AdvancedDecoderOutputAProjectionCudaCandidate::recipe_digest(
  self : Self,
) -> AdvancedDecoderCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn AdvancedDecoderOutputAProjectionCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

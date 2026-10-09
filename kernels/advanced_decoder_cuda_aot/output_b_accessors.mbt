///|
pub fn AdvancedDecoderOutputBProjectionCudaCandidate::family(
  self : Self,
) -> AdvancedDecoderOutputBProjectionFamily {
  self.family
}

///|
pub fn AdvancedDecoderOutputBProjectionCudaCandidate::numerical_contract(
  self : Self,
) -> AdvancedDecoderOutputBProjectionNumericalContract {
  self.numerical_contract
}

///|
pub fn AdvancedDecoderOutputBProjectionCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn AdvancedDecoderOutputBProjectionCudaCandidate::requirements_digest(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn AdvancedDecoderOutputBProjectionCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn AdvancedDecoderOutputBProjectionCudaCandidate::operation(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation {
  self.operation
}

///|
pub fn AdvancedDecoderOutputBProjectionCudaCandidate::capability(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderAotCapability {
  self.capability
}

///|
pub fn AdvancedDecoderOutputBProjectionCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn AdvancedDecoderOutputBProjectionCudaCandidate::maximum_tokens(
  self : Self,
) -> Int {
  self.maximum_tokens
}

///|
pub fn AdvancedDecoderOutputBProjectionCudaCandidate::input_width(
  self : Self,
) -> Int {
  self.input_width
}

///|
pub fn AdvancedDecoderOutputBProjectionCudaCandidate::output_width(
  self : Self,
) -> Int {
  self.output_width
}

///|
pub fn AdvancedDecoderOutputBProjectionCudaCandidate::block_width(
  self : Self,
) -> Int {
  self.block_width
}

///|
pub fn AdvancedDecoderOutputBProjectionCudaCandidate::tensor_parallel_world_size(
  _self : Self,
) -> Int {
  1
}

///|
pub fn AdvancedDecoderOutputBProjectionCudaCandidate::owns_collective(
  _self : Self,
) -> Bool {
  false
}

///|
pub fn AdvancedDecoderOutputBProjectionCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn AdvancedDecoderOutputBProjectionCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[
  @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
] {
  self.operands
}

///|
pub fn AdvancedDecoderOutputBProjectionCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn AdvancedDecoderOutputBProjectionCudaCandidate::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn AdvancedDecoderOutputBProjectionCudaCandidate::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

///|
pub fn AdvancedDecoderOutputBProjectionCudaCandidate::source_digest(
  self : Self,
) -> AdvancedDecoderCudaSourceDigest {
  self.source_digest
}

///|
pub fn AdvancedDecoderOutputBProjectionCudaCandidate::recipe_bytes(
  self : Self,
) -> Bytes {
  self.recipe
}

///|
pub fn AdvancedDecoderOutputBProjectionCudaCandidate::recipe_digest(
  self : Self,
) -> AdvancedDecoderCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn AdvancedDecoderOutputBProjectionCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

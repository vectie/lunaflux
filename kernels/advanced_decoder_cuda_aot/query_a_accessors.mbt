///|
pub fn AdvancedDecoderQueryAProjectionCudaCandidate::family(
  self : Self,
) -> AdvancedDecoderQueryAProjectionFamily {
  self.family
}

///|
pub fn AdvancedDecoderQueryAProjectionCudaCandidate::numerical_contract(
  self : Self,
) -> AdvancedDecoderQueryAProjectionNumericalContract {
  self.numerical_contract
}

///|
pub fn AdvancedDecoderQueryAProjectionCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn AdvancedDecoderQueryAProjectionCudaCandidate::requirements_digest(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn AdvancedDecoderQueryAProjectionCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn AdvancedDecoderQueryAProjectionCudaCandidate::operation(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation {
  self.operation
}

///|
pub fn AdvancedDecoderQueryAProjectionCudaCandidate::capability(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderAotCapability {
  self.capability
}

///|
pub fn AdvancedDecoderQueryAProjectionCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn AdvancedDecoderQueryAProjectionCudaCandidate::maximum_tokens(
  self : Self,
) -> Int {
  self.maximum_tokens
}

///|
pub fn AdvancedDecoderQueryAProjectionCudaCandidate::input_width(
  self : Self,
) -> Int {
  self.input_width
}

///|
pub fn AdvancedDecoderQueryAProjectionCudaCandidate::output_width(
  self : Self,
) -> Int {
  self.output_width
}

///|
pub fn AdvancedDecoderQueryAProjectionCudaCandidate::block_width(
  self : Self,
) -> Int {
  self.block_width
}

///|
pub fn AdvancedDecoderQueryAProjectionCudaCandidate::tensor_parallel_replicated(
  _self : Self,
) -> Bool {
  true
}

///|
pub fn AdvancedDecoderQueryAProjectionCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn AdvancedDecoderQueryAProjectionCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[
  @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
] {
  self.operands
}

///|
pub fn AdvancedDecoderQueryAProjectionCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn AdvancedDecoderQueryAProjectionCudaCandidate::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn AdvancedDecoderQueryAProjectionCudaCandidate::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

///|
pub fn AdvancedDecoderQueryAProjectionCudaCandidate::source_digest(
  self : Self,
) -> AdvancedDecoderCudaSourceDigest {
  self.source_digest
}

///|
pub fn AdvancedDecoderQueryAProjectionCudaCandidate::recipe_bytes(
  self : Self,
) -> Bytes {
  self.recipe
}

///|
pub fn AdvancedDecoderQueryAProjectionCudaCandidate::recipe_digest(
  self : Self,
) -> AdvancedDecoderCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn AdvancedDecoderQueryAProjectionCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

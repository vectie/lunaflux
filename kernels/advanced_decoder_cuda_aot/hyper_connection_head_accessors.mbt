///|
pub fn AdvancedDecoderHyperConnectionHeadReduceCudaCandidate::family(
  self : Self,
) -> AdvancedDecoderHyperConnectionAotFamily {
  self.family
}

///|
pub fn AdvancedDecoderHyperConnectionHeadReduceCudaCandidate::numerical_contract(
  self : Self,
) -> AdvancedDecoderHyperConnectionNumericalContract {
  self.numerical_contract
}

///|
pub fn AdvancedDecoderHyperConnectionHeadReduceCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn AdvancedDecoderHyperConnectionHeadReduceCudaCandidate::requirements_digest(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn AdvancedDecoderHyperConnectionHeadReduceCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn AdvancedDecoderHyperConnectionHeadReduceCudaCandidate::operation(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation {
  self.operation
}

///|
pub fn AdvancedDecoderHyperConnectionHeadReduceCudaCandidate::capability(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderAotCapability {
  self.capability
}

///|
pub fn AdvancedDecoderHyperConnectionHeadReduceCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn AdvancedDecoderHyperConnectionHeadReduceCudaCandidate::maximum_tokens(
  self : Self,
) -> Int {
  self.maximum_tokens
}

///|
pub fn AdvancedDecoderHyperConnectionHeadReduceCudaCandidate::stream_width(
  self : Self,
) -> Int {
  self.stream_width
}

///|
pub fn AdvancedDecoderHyperConnectionHeadReduceCudaCandidate::hidden_width(
  self : Self,
) -> Int {
  self.hidden_width
}

///|
pub fn AdvancedDecoderHyperConnectionHeadReduceCudaCandidate::control_epsilon(
  self : Self,
) -> Double {
  self.control_epsilon
}

///|
pub fn AdvancedDecoderHyperConnectionHeadReduceCudaCandidate::normalization_epsilon(
  self : Self,
) -> Double {
  self.normalization_epsilon
}

///|
pub fn AdvancedDecoderHyperConnectionHeadReduceCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn AdvancedDecoderHyperConnectionHeadReduceCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[
  @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
] {
  self.operands
}

///|
pub fn AdvancedDecoderHyperConnectionHeadReduceCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn AdvancedDecoderHyperConnectionHeadReduceCudaCandidate::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn AdvancedDecoderHyperConnectionHeadReduceCudaCandidate::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

///|
pub fn AdvancedDecoderHyperConnectionHeadReduceCudaCandidate::source_digest(
  self : Self,
) -> AdvancedDecoderCudaSourceDigest {
  self.source_digest
}

///|
pub fn AdvancedDecoderHyperConnectionHeadReduceCudaCandidate::recipe_bytes(
  self : Self,
) -> Bytes {
  self.recipe
}

///|
pub fn AdvancedDecoderHyperConnectionHeadReduceCudaCandidate::recipe_digest(
  self : Self,
) -> AdvancedDecoderCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn AdvancedDecoderHyperConnectionHeadReduceCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

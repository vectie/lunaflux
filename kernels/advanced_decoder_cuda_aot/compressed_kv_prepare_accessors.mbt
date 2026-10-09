///|
pub fn AdvancedDecoderCompressedKvPrepareCudaCandidate::family(
  self : Self,
) -> AdvancedDecoderCompressedKvPrepareAotFamily {
  self.family
}

///|
pub fn AdvancedDecoderCompressedKvPrepareCudaCandidate::numerical_contract(
  self : Self,
) -> AdvancedDecoderCompressedKvPrepareNumericalContract {
  self.numerical_contract
}

///|
pub fn AdvancedDecoderCompressedKvPrepareCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn AdvancedDecoderCompressedKvPrepareCudaCandidate::requirements_digest(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn AdvancedDecoderCompressedKvPrepareCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn AdvancedDecoderCompressedKvPrepareCudaCandidate::operation(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation {
  self.operation
}

///|
pub fn AdvancedDecoderCompressedKvPrepareCudaCandidate::capability(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderAotCapability {
  self.capability
}

///|
pub fn AdvancedDecoderCompressedKvPrepareCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn AdvancedDecoderCompressedKvPrepareCudaCandidate::maximum_batches(
  self : Self,
) -> Int {
  self.maximum_batches
}

///|
pub fn AdvancedDecoderCompressedKvPrepareCudaCandidate::window_size(
  self : Self,
) -> Int {
  self.window_size
}

///|
pub fn AdvancedDecoderCompressedKvPrepareCudaCandidate::compression_ratio(
  self : Self,
) -> Int {
  self.compression_ratio
}

///|
pub fn AdvancedDecoderCompressedKvPrepareCudaCandidate::head_dimension(
  self : Self,
) -> Int {
  self.head_dimension
}

///|
pub fn AdvancedDecoderCompressedKvPrepareCudaCandidate::maximum_position_count(
  self : Self,
) -> Int {
  self.maximum_position_count
}

///|
pub fn AdvancedDecoderCompressedKvPrepareCudaCandidate::maximum_compressed_slots(
  self : Self,
) -> Int {
  self.maximum_compressed_slots
}

///|
pub fn AdvancedDecoderCompressedKvPrepareCudaCandidate::maximum_key_slots(
  self : Self,
) -> Int {
  self.maximum_key_slots
}

///|
pub fn AdvancedDecoderCompressedKvPrepareCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn AdvancedDecoderCompressedKvPrepareCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[
  @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
] {
  self.operands
}

///|
pub fn AdvancedDecoderCompressedKvPrepareCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn AdvancedDecoderCompressedKvPrepareCudaCandidate::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn AdvancedDecoderCompressedKvPrepareCudaCandidate::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

///|
pub fn AdvancedDecoderCompressedKvPrepareCudaCandidate::source_digest(
  self : Self,
) -> AdvancedDecoderCudaSourceDigest {
  self.source_digest
}

///|
pub fn AdvancedDecoderCompressedKvPrepareCudaCandidate::recipe_bytes(
  self : Self,
) -> Bytes {
  self.recipe
}

///|
pub fn AdvancedDecoderCompressedKvPrepareCudaCandidate::recipe_digest(
  self : Self,
) -> AdvancedDecoderCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn AdvancedDecoderCompressedKvPrepareCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

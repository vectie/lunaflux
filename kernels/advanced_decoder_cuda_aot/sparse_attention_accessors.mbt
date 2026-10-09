///|
pub fn AdvancedDecoderSparseAttentionCudaCandidate::family(
  self : Self,
) -> AdvancedDecoderSparseAttentionAotFamily {
  self.family
}

///|
pub fn AdvancedDecoderSparseAttentionCudaCandidate::numerical_contract(
  self : Self,
) -> AdvancedDecoderSparseAttentionNumericalContract {
  self.numerical_contract
}

///|
pub fn AdvancedDecoderSparseAttentionCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn AdvancedDecoderSparseAttentionCudaCandidate::requirements_digest(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn AdvancedDecoderSparseAttentionCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn AdvancedDecoderSparseAttentionCudaCandidate::operation(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation {
  self.operation
}

///|
pub fn AdvancedDecoderSparseAttentionCudaCandidate::capability(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderAotCapability {
  self.capability
}

///|
pub fn AdvancedDecoderSparseAttentionCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn AdvancedDecoderSparseAttentionCudaCandidate::maximum_tokens(
  self : Self,
) -> Int {
  self.maximum_tokens
}

///|
pub fn AdvancedDecoderSparseAttentionCudaCandidate::maximum_batches(
  self : Self,
) -> Int {
  self.maximum_batches
}

///|
pub fn AdvancedDecoderSparseAttentionCudaCandidate::query_head_count(
  self : Self,
) -> Int {
  self.query_head_count
}

///|
pub fn AdvancedDecoderSparseAttentionCudaCandidate::head_dimension(
  self : Self,
) -> Int {
  self.head_dimension
}

///|
pub fn AdvancedDecoderSparseAttentionCudaCandidate::window_size(
  self : Self,
) -> Int {
  self.window_size
}

///|
pub fn AdvancedDecoderSparseAttentionCudaCandidate::compression_ratio(
  self : Self,
) -> Int {
  self.compression_ratio
}

///|
pub fn AdvancedDecoderSparseAttentionCudaCandidate::selected_width(
  self : Self,
) -> Int {
  self.selected_width
}

///|
pub fn AdvancedDecoderSparseAttentionCudaCandidate::maximum_key_slots(
  self : Self,
) -> Int {
  self.maximum_key_slots
}

///|
pub fn AdvancedDecoderSparseAttentionCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn AdvancedDecoderSparseAttentionCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[
  @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
] {
  self.operands
}

///|
pub fn AdvancedDecoderSparseAttentionCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn AdvancedDecoderSparseAttentionCudaCandidate::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn AdvancedDecoderSparseAttentionCudaCandidate::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

///|
pub fn AdvancedDecoderSparseAttentionCudaCandidate::source_digest(
  self : Self,
) -> AdvancedDecoderCudaSourceDigest {
  self.source_digest
}

///|
pub fn AdvancedDecoderSparseAttentionCudaCandidate::recipe_bytes(
  self : Self,
) -> Bytes {
  self.recipe
}

///|
pub fn AdvancedDecoderSparseAttentionCudaCandidate::recipe_digest(
  self : Self,
) -> AdvancedDecoderCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn AdvancedDecoderSparseAttentionCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

///|
pub fn AdvancedDecoderHyperConnectionCudaCandidate::family(
  self : Self,
) -> AdvancedDecoderHyperConnectionAotFamily {
  self.family
}

///|
pub fn AdvancedDecoderHyperConnectionCudaCandidate::numerical_contract(
  self : Self,
) -> AdvancedDecoderHyperConnectionNumericalContract {
  self.numerical_contract
}

///|
pub fn AdvancedDecoderHyperConnectionCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn AdvancedDecoderHyperConnectionCudaCandidate::requirements_digest(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn AdvancedDecoderHyperConnectionCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn AdvancedDecoderHyperConnectionCudaCandidate::operation(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation {
  self.operation
}

///|
pub fn AdvancedDecoderHyperConnectionCudaCandidate::capability(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderAotCapability {
  self.capability
}

///|
pub fn AdvancedDecoderHyperConnectionCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn AdvancedDecoderHyperConnectionCudaCandidate::maximum_tokens(
  self : Self,
) -> Int {
  self.maximum_tokens
}

///|
pub fn AdvancedDecoderHyperConnectionCudaCandidate::stream_width(
  self : Self,
) -> Int {
  self.stream_width
}

///|
pub fn AdvancedDecoderHyperConnectionCudaCandidate::hidden_width(
  self : Self,
) -> Int {
  self.hidden_width
}

///|
pub fn AdvancedDecoderHyperConnectionCudaCandidate::sinkhorn_iterations(
  self : Self,
) -> Int {
  self.sinkhorn_iterations
}

///|
pub fn AdvancedDecoderHyperConnectionCudaCandidate::control_epsilon(
  self : Self,
) -> Double {
  self.control_epsilon
}

///|
pub fn AdvancedDecoderHyperConnectionCudaCandidate::normalization_epsilon(
  self : Self,
) -> Double {
  self.normalization_epsilon
}

///|
pub fn AdvancedDecoderHyperConnectionCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn AdvancedDecoderHyperConnectionCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[
  @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
] {
  self.operands
}

///|
pub fn AdvancedDecoderHyperConnectionCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn AdvancedDecoderHyperConnectionCudaCandidate::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn AdvancedDecoderHyperConnectionCudaCandidate::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

///|
pub fn AdvancedDecoderHyperConnectionCudaCandidate::source_digest(
  self : Self,
) -> AdvancedDecoderCudaSourceDigest {
  self.source_digest
}

///|
pub fn AdvancedDecoderHyperConnectionCudaCandidate::recipe_bytes(
  self : Self,
) -> Bytes {
  self.recipe
}

///|
pub fn AdvancedDecoderHyperConnectionCudaCandidate::recipe_digest(
  self : Self,
) -> AdvancedDecoderCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn AdvancedDecoderHyperConnectionCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

///|
pub fn AdvancedDecoderHyperConnectionPreReduceCudaCandidate::family(
  self : Self,
) -> AdvancedDecoderHyperConnectionAotFamily {
  self.family
}

///|
pub fn AdvancedDecoderHyperConnectionPreReduceCudaCandidate::numerical_contract(
  self : Self,
) -> AdvancedDecoderHyperConnectionNumericalContract {
  self.numerical_contract
}

///|
pub fn AdvancedDecoderHyperConnectionPreReduceCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn AdvancedDecoderHyperConnectionPreReduceCudaCandidate::requirements_digest(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn AdvancedDecoderHyperConnectionPreReduceCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn AdvancedDecoderHyperConnectionPreReduceCudaCandidate::operation(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation {
  self.operation
}

///|
pub fn AdvancedDecoderHyperConnectionPreReduceCudaCandidate::capability(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderAotCapability {
  self.capability
}

///|
pub fn AdvancedDecoderHyperConnectionPreReduceCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn AdvancedDecoderHyperConnectionPreReduceCudaCandidate::maximum_tokens(
  self : Self,
) -> Int {
  self.maximum_tokens
}

///|
pub fn AdvancedDecoderHyperConnectionPreReduceCudaCandidate::stream_width(
  self : Self,
) -> Int {
  self.stream_width
}

///|
pub fn AdvancedDecoderHyperConnectionPreReduceCudaCandidate::hidden_width(
  self : Self,
) -> Int {
  self.hidden_width
}

///|
pub fn AdvancedDecoderHyperConnectionPreReduceCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn AdvancedDecoderHyperConnectionPreReduceCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[
  @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
] {
  self.operands
}

///|
pub fn AdvancedDecoderHyperConnectionPreReduceCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn AdvancedDecoderHyperConnectionPreReduceCudaCandidate::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn AdvancedDecoderHyperConnectionPreReduceCudaCandidate::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

///|
pub fn AdvancedDecoderHyperConnectionPreReduceCudaCandidate::source_digest(
  self : Self,
) -> AdvancedDecoderCudaSourceDigest {
  self.source_digest
}

///|
pub fn AdvancedDecoderHyperConnectionPreReduceCudaCandidate::recipe_bytes(
  self : Self,
) -> Bytes {
  self.recipe
}

///|
pub fn AdvancedDecoderHyperConnectionPreReduceCudaCandidate::recipe_digest(
  self : Self,
) -> AdvancedDecoderCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn AdvancedDecoderHyperConnectionPreReduceCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

///|
pub fn AdvancedDecoderHyperConnectionPostCombineCudaCandidate::family(
  self : Self,
) -> AdvancedDecoderHyperConnectionAotFamily {
  self.family
}

///|
pub fn AdvancedDecoderHyperConnectionPostCombineCudaCandidate::numerical_contract(
  self : Self,
) -> AdvancedDecoderHyperConnectionNumericalContract {
  self.numerical_contract
}

///|
pub fn AdvancedDecoderHyperConnectionPostCombineCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn AdvancedDecoderHyperConnectionPostCombineCudaCandidate::requirements_digest(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn AdvancedDecoderHyperConnectionPostCombineCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn AdvancedDecoderHyperConnectionPostCombineCudaCandidate::operation(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation {
  self.operation
}

///|
pub fn AdvancedDecoderHyperConnectionPostCombineCudaCandidate::capability(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderAotCapability {
  self.capability
}

///|
pub fn AdvancedDecoderHyperConnectionPostCombineCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn AdvancedDecoderHyperConnectionPostCombineCudaCandidate::maximum_tokens(
  self : Self,
) -> Int {
  self.maximum_tokens
}

///|
pub fn AdvancedDecoderHyperConnectionPostCombineCudaCandidate::stream_width(
  self : Self,
) -> Int {
  self.stream_width
}

///|
pub fn AdvancedDecoderHyperConnectionPostCombineCudaCandidate::hidden_width(
  self : Self,
) -> Int {
  self.hidden_width
}

///|
pub fn AdvancedDecoderHyperConnectionPostCombineCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn AdvancedDecoderHyperConnectionPostCombineCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[
  @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
] {
  self.operands
}

///|
pub fn AdvancedDecoderHyperConnectionPostCombineCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn AdvancedDecoderHyperConnectionPostCombineCudaCandidate::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn AdvancedDecoderHyperConnectionPostCombineCudaCandidate::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

///|
pub fn AdvancedDecoderHyperConnectionPostCombineCudaCandidate::source_digest(
  self : Self,
) -> AdvancedDecoderCudaSourceDigest {
  self.source_digest
}

///|
pub fn AdvancedDecoderHyperConnectionPostCombineCudaCandidate::recipe_bytes(
  self : Self,
) -> Bytes {
  self.recipe
}

///|
pub fn AdvancedDecoderHyperConnectionPostCombineCudaCandidate::recipe_digest(
  self : Self,
) -> AdvancedDecoderCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn AdvancedDecoderHyperConnectionPostCombineCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

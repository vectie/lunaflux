///|
pub fn JointDiffusionAttentionCudaStage::scope(
  self : Self,
) -> JointDiffusionAttentionRowScope {
  self.scope
}

///|
pub fn JointDiffusionAttentionCudaStage::kind(
  self : Self,
) -> JointDiffusionAttentionStageKind {
  self.kind
}

///|
pub fn JointDiffusionAttentionCudaStage::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn JointDiffusionAttentionCudaStage::operands(
  self : Self,
) -> ReadOnlyArray[
  @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
] {
  self.operands
}

///|
pub fn JointDiffusionTransformerAttentionCudaCandidate::numerical_contract(
  self : Self,
) -> JointDiffusionCudaNumericalContract {
  self.numerical_contract
}

///|
pub fn JointDiffusionTransformerAttentionCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn JointDiffusionTransformerAttentionCudaCandidate::requirements_digest(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn JointDiffusionTransformerAttentionCudaCandidate::request_digest(
  self : Self,
) -> @spec.PlanDigest {
  self.request_digest
}

///|
pub fn JointDiffusionTransformerAttentionCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn JointDiffusionTransformerAttentionCudaCandidate::operation(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation {
  self.operation
}

///|
pub fn JointDiffusionTransformerAttentionCudaCandidate::capability(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionAotCapability {
  self.capability
}

///|
pub fn JointDiffusionTransformerAttentionCudaCandidate::layer_count(
  self : Self,
) -> Int {
  self.layer_count
}

///|
pub fn JointDiffusionTransformerAttentionCudaCandidate::refiner_layer_count(
  self : Self,
) -> Int {
  self.refiner_layer_count
}

///|
pub fn JointDiffusionTransformerAttentionCudaCandidate::packed_row_count(
  self : Self,
) -> Int {
  self.packed_row_count
}

///|
pub fn JointDiffusionTransformerAttentionCudaCandidate::refiner_row_count(
  self : Self,
) -> Int {
  self.refiner_row_count
}

///|
pub fn JointDiffusionTransformerAttentionCudaCandidate::hidden_size(
  self : Self,
) -> Int {
  self.hidden_size
}

///|
pub fn JointDiffusionTransformerAttentionCudaCandidate::head_count(
  self : Self,
) -> Int {
  self.head_count
}

///|
pub fn JointDiffusionTransformerAttentionCudaCandidate::head_dimension(
  self : Self,
) -> Int {
  self.head_dimension
}

///|
pub fn JointDiffusionTransformerAttentionCudaCandidate::inner_size(
  self : Self,
) -> Int {
  self.inner_size
}

///|
pub fn JointDiffusionTransformerAttentionCudaCandidate::rotary_dimension(
  self : Self,
) -> Int {
  self.rotary_dimension
}

///|
pub fn JointDiffusionTransformerAttentionCudaCandidate::qk_epsilon_numerator(
  self : Self,
) -> Int {
  self.qk_epsilon_numerator
}

///|
pub fn JointDiffusionTransformerAttentionCudaCandidate::qk_epsilon_denominator(
  self : Self,
) -> Int {
  self.qk_epsilon_denominator
}

///|
pub fn JointDiffusionTransformerAttentionCudaCandidate::projection_weight_element_count(
  self : Self,
) -> Int64 {
  self.projection_weight_element_count
}

///|
pub fn JointDiffusionTransformerAttentionCudaCandidate::norm_weight_element_count(
  self : Self,
) -> Int64 {
  self.norm_weight_element_count
}

///|
pub fn JointDiffusionTransformerAttentionCudaCandidate::stages(
  self : Self,
) -> ReadOnlyArray[JointDiffusionAttentionCudaStage] {
  self.stages
}

///|
pub fn JointDiffusionTransformerAttentionCudaCandidate::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

///|
pub fn JointDiffusionTransformerAttentionCudaCandidate::source_digest(
  self : Self,
) -> JointDiffusionCudaSourceDigest {
  self.source_digest
}

///|
pub fn JointDiffusionTransformerAttentionCudaCandidate::recipe_bytes(
  self : Self,
) -> Bytes {
  self.recipe
}

///|
pub fn JointDiffusionTransformerAttentionCudaCandidate::recipe_digest(
  self : Self,
) -> JointDiffusionCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn JointDiffusionTransformerAttentionCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

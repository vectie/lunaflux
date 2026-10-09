///|
pub fn JointDiffusionTransformerMlpCudaCandidate::numerical_contract(
  self : Self,
) -> JointDiffusionCudaNumericalContract {
  self.numerical_contract
}

///|
pub fn JointDiffusionTransformerMlpCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn JointDiffusionTransformerMlpCudaCandidate::requirements_digest(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn JointDiffusionTransformerMlpCudaCandidate::request_digest(
  self : Self,
) -> @spec.PlanDigest {
  self.request_digest
}

///|
pub fn JointDiffusionTransformerMlpCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn JointDiffusionTransformerMlpCudaCandidate::operation(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation {
  self.operation
}

///|
pub fn JointDiffusionTransformerMlpCudaCandidate::capability(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionAotCapability {
  self.capability
}

///|
pub fn JointDiffusionTransformerMlpCudaCandidate::layer_count(
  self : Self,
) -> Int {
  self.layer_count
}

///|
pub fn JointDiffusionTransformerMlpCudaCandidate::refiner_layer_count(
  self : Self,
) -> Int {
  self.refiner_layer_count
}

///|
pub fn JointDiffusionTransformerMlpCudaCandidate::packed_row_count(
  self : Self,
) -> Int {
  self.packed_row_count
}

///|
pub fn JointDiffusionTransformerMlpCudaCandidate::refiner_row_count(
  self : Self,
) -> Int {
  self.refiner_row_count
}

///|
pub fn JointDiffusionTransformerMlpCudaCandidate::hidden_size(
  self : Self,
) -> Int {
  self.hidden_size
}

///|
pub fn JointDiffusionTransformerMlpCudaCandidate::intermediate_size(
  self : Self,
) -> Int {
  self.intermediate_size
}

///|
pub fn JointDiffusionTransformerMlpCudaCandidate::packed_weight_element_count(
  self : Self,
) -> Int64 {
  self.packed_weight_element_count
}

///|
pub fn JointDiffusionTransformerMlpCudaCandidate::down_weight_element_count(
  self : Self,
) -> Int64 {
  self.down_weight_element_count
}

///|
pub fn JointDiffusionTransformerMlpCudaCandidate::packed_dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.packed_dimensions
}

///|
pub fn JointDiffusionTransformerMlpCudaCandidate::refiner_dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.refiner_dimensions
}

///|
pub fn JointDiffusionTransformerMlpCudaCandidate::packed_operands(
  self : Self,
) -> ReadOnlyArray[
  @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
] {
  self.packed_operands
}

///|
pub fn JointDiffusionTransformerMlpCudaCandidate::refiner_operands(
  self : Self,
) -> ReadOnlyArray[
  @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
] {
  self.refiner_operands
}

///|
pub fn JointDiffusionTransformerMlpCudaCandidate::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

///|
pub fn JointDiffusionTransformerMlpCudaCandidate::source_digest(
  self : Self,
) -> JointDiffusionCudaSourceDigest {
  self.source_digest
}

///|
pub fn JointDiffusionTransformerMlpCudaCandidate::recipe_bytes(
  self : Self,
) -> Bytes {
  self.recipe
}

///|
pub fn JointDiffusionTransformerMlpCudaCandidate::recipe_digest(
  self : Self,
) -> JointDiffusionCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn JointDiffusionTransformerMlpCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

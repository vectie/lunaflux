///|
pub fn JointDiffusionTimestepMlpCudaCandidate::numerical_contract(
  self : Self,
) -> JointDiffusionCudaNumericalContract {
  self.numerical_contract
}

///|
pub fn JointDiffusionTimestepMlpCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn JointDiffusionTimestepMlpCudaCandidate::requirements_digest(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn JointDiffusionTimestepMlpCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn JointDiffusionTimestepMlpCudaCandidate::operation(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation {
  self.operation
}

///|
pub fn JointDiffusionTimestepMlpCudaCandidate::capability(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionAotCapability {
  self.capability
}

///|
pub fn JointDiffusionTimestepMlpCudaCandidate::values_digest(
  self : Self,
) -> @spec.PlanDigest {
  self.values_digest
}

///|
pub fn JointDiffusionTimestepMlpCudaCandidate::row_count(self : Self) -> Int {
  self.row_count
}

///|
pub fn JointDiffusionTimestepMlpCudaCandidate::input_width(self : Self) -> Int {
  self.input_width
}

///|
pub fn JointDiffusionTimestepMlpCudaCandidate::hidden_width(self : Self) -> Int {
  self.hidden_width
}

///|
pub fn JointDiffusionTimestepMlpCudaCandidate::output_width(self : Self) -> Int {
  self.output_width
}

///|
pub fn JointDiffusionTimestepMlpCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn JointDiffusionTimestepMlpCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[
  @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
] {
  self.operands
}

///|
pub fn JointDiffusionTimestepMlpCudaCandidate::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

///|
pub fn JointDiffusionTimestepMlpCudaCandidate::source_digest(
  self : Self,
) -> JointDiffusionCudaSourceDigest {
  self.source_digest
}

///|
pub fn JointDiffusionTimestepMlpCudaCandidate::recipe_bytes(
  self : Self,
) -> Bytes {
  self.recipe
}

///|
pub fn JointDiffusionTimestepMlpCudaCandidate::recipe_digest(
  self : Self,
) -> JointDiffusionCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn JointDiffusionTimestepMlpCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

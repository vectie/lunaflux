///|
pub fn JointDiffusionTimestepFrequencyCudaCandidate::numerical_contract(
  self : Self,
) -> JointDiffusionCudaNumericalContract {
  self.numerical_contract
}

///|
pub fn JointDiffusionTimestepFrequencyCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn JointDiffusionTimestepFrequencyCudaCandidate::requirements_digest(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn JointDiffusionTimestepFrequencyCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn JointDiffusionTimestepFrequencyCudaCandidate::operation(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation {
  self.operation
}

///|
pub fn JointDiffusionTimestepFrequencyCudaCandidate::capability(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionAotCapability {
  self.capability
}

///|
pub fn JointDiffusionTimestepFrequencyCudaCandidate::values_digest(
  self : Self,
) -> @spec.PlanDigest {
  self.values_digest
}

///|
pub fn JointDiffusionTimestepFrequencyCudaCandidate::row_count(
  self : Self,
) -> Int {
  self.row_count
}

///|
pub fn JointDiffusionTimestepFrequencyCudaCandidate::frequency_width(
  self : Self,
) -> Int {
  self.frequency_width
}

///|
pub fn JointDiffusionTimestepFrequencyCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn JointDiffusionTimestepFrequencyCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[
  @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
] {
  self.operands
}

///|
pub fn JointDiffusionTimestepFrequencyCudaCandidate::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

///|
pub fn JointDiffusionTimestepFrequencyCudaCandidate::source_digest(
  self : Self,
) -> JointDiffusionCudaSourceDigest {
  self.source_digest
}

///|
pub fn JointDiffusionTimestepFrequencyCudaCandidate::recipe_bytes(
  self : Self,
) -> Bytes {
  self.recipe
}

///|
pub fn JointDiffusionTimestepFrequencyCudaCandidate::recipe_digest(
  self : Self,
) -> JointDiffusionCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn JointDiffusionTimestepFrequencyCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

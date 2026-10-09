///|
pub fn JointDiffusionDistinctTimestepAdaLnCudaCandidate::numerical_contract(
  self : Self,
) -> JointDiffusionCudaNumericalContract {
  self.numerical_contract
}

///|
pub fn JointDiffusionDistinctTimestepAdaLnCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn JointDiffusionDistinctTimestepAdaLnCudaCandidate::requirements_digest(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn JointDiffusionDistinctTimestepAdaLnCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn JointDiffusionDistinctTimestepAdaLnCudaCandidate::operation(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation {
  self.operation
}

///|
pub fn JointDiffusionDistinctTimestepAdaLnCudaCandidate::capability(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionAotCapability {
  self.capability
}

///|
pub fn JointDiffusionDistinctTimestepAdaLnCudaCandidate::values_digest(
  self : Self,
) -> @spec.PlanDigest {
  self.values_digest
}

///|
pub fn JointDiffusionDistinctTimestepAdaLnCudaCandidate::row_count(
  self : Self,
) -> Int {
  self.row_count
}

///|
pub fn JointDiffusionDistinctTimestepAdaLnCudaCandidate::input_width(
  self : Self,
) -> Int {
  self.input_width
}

///|
pub fn JointDiffusionDistinctTimestepAdaLnCudaCandidate::hidden_size(
  self : Self,
) -> Int {
  self.hidden_size
}

///|
pub fn JointDiffusionDistinctTimestepAdaLnCudaCandidate::weight_element_count(
  self : Self,
) -> Int64 {
  self.weight_element_count
}

///|
pub fn JointDiffusionDistinctTimestepAdaLnCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn JointDiffusionDistinctTimestepAdaLnCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[
  @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
] {
  self.operands
}

///|
pub fn JointDiffusionDistinctTimestepAdaLnCudaCandidate::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

///|
pub fn JointDiffusionDistinctTimestepAdaLnCudaCandidate::source_digest(
  self : Self,
) -> JointDiffusionCudaSourceDigest {
  self.source_digest
}

///|
pub fn JointDiffusionDistinctTimestepAdaLnCudaCandidate::recipe_bytes(
  self : Self,
) -> Bytes {
  self.recipe
}

///|
pub fn JointDiffusionDistinctTimestepAdaLnCudaCandidate::recipe_digest(
  self : Self,
) -> JointDiffusionCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn JointDiffusionDistinctTimestepAdaLnCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

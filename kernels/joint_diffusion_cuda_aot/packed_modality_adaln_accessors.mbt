///|
pub fn JointDiffusionPackedModalityAdaLnCudaCandidate::numerical_contract(
  self : Self,
) -> JointDiffusionCudaNumericalContract {
  self.numerical_contract
}

///|
pub fn JointDiffusionPackedModalityAdaLnCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn JointDiffusionPackedModalityAdaLnCudaCandidate::requirements_digest(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn JointDiffusionPackedModalityAdaLnCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn JointDiffusionPackedModalityAdaLnCudaCandidate::operation(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation {
  self.operation
}

///|
pub fn JointDiffusionPackedModalityAdaLnCudaCandidate::capability(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionAotCapability {
  self.capability
}

///|
pub fn JointDiffusionPackedModalityAdaLnCudaCandidate::timestep_digest(
  self : Self,
) -> @spec.PlanDigest {
  self.timestep_digest
}

///|
pub fn JointDiffusionPackedModalityAdaLnCudaCandidate::modality_digest(
  self : Self,
) -> @spec.PlanDigest {
  self.modality_digest
}

///|
pub fn JointDiffusionPackedModalityAdaLnCudaCandidate::hidden_size(
  self : Self,
) -> Int {
  self.hidden_size
}

///|
pub fn JointDiffusionPackedModalityAdaLnCudaCandidate::packed_row_count(
  self : Self,
) -> Int {
  self.packed_rows
}

///|
pub fn JointDiffusionPackedModalityAdaLnCudaCandidate::video_row_count(
  self : Self,
) -> Int {
  self.video_rows
}

///|
pub fn JointDiffusionPackedModalityAdaLnCudaCandidate::audio_row_count(
  self : Self,
) -> Int {
  self.audio_rows
}

///|
pub fn JointDiffusionPackedModalityAdaLnCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn JointDiffusionPackedModalityAdaLnCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[
  @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
] {
  self.operands
}

///|
pub fn JointDiffusionPackedModalityAdaLnCudaCandidate::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

///|
pub fn JointDiffusionPackedModalityAdaLnCudaCandidate::source_digest(
  self : Self,
) -> JointDiffusionCudaSourceDigest {
  self.source_digest
}

///|
pub fn JointDiffusionPackedModalityAdaLnCudaCandidate::recipe_bytes(
  self : Self,
) -> Bytes {
  self.recipe
}

///|
pub fn JointDiffusionPackedModalityAdaLnCudaCandidate::recipe_digest(
  self : Self,
) -> JointDiffusionCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn JointDiffusionPackedModalityAdaLnCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

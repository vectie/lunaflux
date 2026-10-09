///|
pub fn JointDiffusionAudioVaeFirstAmpStageCudaCandidate::numerical_contract(
  self : Self,
) -> JointDiffusionCudaNumericalContract {
  self.numerical_contract
}

///|
pub fn JointDiffusionAudioVaeFirstAmpStageCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn JointDiffusionAudioVaeFirstAmpStageCudaCandidate::requirements_digest(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn JointDiffusionAudioVaeFirstAmpStageCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn JointDiffusionAudioVaeFirstAmpStageCudaCandidate::operation(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation {
  self.operation
}

///|
pub fn JointDiffusionAudioVaeFirstAmpStageCudaCandidate::capability(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionAotCapability {
  self.capability
}

///|
pub fn JointDiffusionAudioVaeFirstAmpStageCudaCandidate::request_shape(
  self : Self,
) -> @joint_diffusion_plan.JointDiffusionRequestShape {
  self.request_shape
}

///|
pub fn JointDiffusionAudioVaeFirstAmpStageCudaCandidate::parameter_layout_digest(
  self : Self,
) -> @joint_audio_amp_parameter_layout.JointAudioAmpParameterLayoutDigest {
  self.parameter_layout_digest
}

///|
pub fn JointDiffusionAudioVaeFirstAmpStageCudaCandidate::workspace_layout_digest(
  self : Self,
) -> @joint_audio_amp_workspace_layout.JointAudioAmpWorkspaceLayoutDigest {
  self.workspace_layout_digest
}

///|
pub fn JointDiffusionAudioVaeFirstAmpStageCudaCandidate::batch_count(
  self : Self,
) -> Int {
  self.batch_count
}

///|
pub fn JointDiffusionAudioVaeFirstAmpStageCudaCandidate::channel_count(
  self : Self,
) -> Int {
  self.channel_count
}

///|
pub fn JointDiffusionAudioVaeFirstAmpStageCudaCandidate::time_count(
  self : Self,
) -> Int {
  self.time_count
}

///|
pub fn JointDiffusionAudioVaeFirstAmpStageCudaCandidate::element_count(
  self : Self,
) -> Int64 {
  self.element_count
}

///|
pub fn JointDiffusionAudioVaeFirstAmpStageCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn JointDiffusionAudioVaeFirstAmpStageCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[
  @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
] {
  self.operands
}

///|
pub fn JointDiffusionAudioVaeFirstAmpStageCudaCandidate::stages(
  self : Self,
) -> ReadOnlyArray[JointDiffusionAudioVaeFirstAmpExecutionStage] {
  self.stages
}

///|
pub fn JointDiffusionAudioVaeFirstAmpStageCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn JointDiffusionAudioVaeFirstAmpStageCudaCandidate::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

///|
pub fn JointDiffusionAudioVaeFirstAmpStageCudaCandidate::source_digest(
  self : Self,
) -> JointDiffusionCudaSourceDigest {
  self.source_digest
}

///|
pub fn JointDiffusionAudioVaeFirstAmpStageCudaCandidate::recipe_bytes(
  self : Self,
) -> Bytes {
  self.recipe
}

///|
pub fn JointDiffusionAudioVaeFirstAmpStageCudaCandidate::recipe_digest(
  self : Self,
) -> JointDiffusionCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn JointDiffusionAudioVaeFirstAmpStageCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

///|
pub fn JointDiffusionAudioVaeFirstAmpExecutionStage::identity(
  self : Self,
) -> @joint_audio_amp_workspace_layout.JointAudioAmpExecutionStage {
  self.identity
}

///|
pub fn JointDiffusionAudioVaeFirstAmpExecutionStage::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn JointDiffusionAudioVaeFirstAmpExecutionStage::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn JointDiffusionAudioVaeFirstAmpExecutionStage::parameter_region_indices(
  self : Self,
) -> ReadOnlyArray[Int] {
  self.parameter_region_indices
}

///|
pub fn JointDiffusionAudioVaeFirstAmpExecutionStage::workspace_roles(
  self : Self,
) -> ReadOnlyArray[@joint_audio_amp_workspace_layout.JointAudioAmpWorkspaceRole] {
  self.workspace_roles
}

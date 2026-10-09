///|
pub fn JointDiffusionAudioVaeAmpActivationCudaCandidate::numerical_contract(
  self : Self,
) -> JointDiffusionCudaNumericalContract {
  self.numerical_contract
}

///|
pub fn JointDiffusionAudioVaeAmpActivationCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn JointDiffusionAudioVaeAmpActivationCudaCandidate::requirements_digest(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn JointDiffusionAudioVaeAmpActivationCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn JointDiffusionAudioVaeAmpActivationCudaCandidate::operation(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation {
  self.operation
}

///|
pub fn JointDiffusionAudioVaeAmpActivationCudaCandidate::capability(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionAotCapability {
  self.capability
}

///|
pub fn JointDiffusionAudioVaeAmpActivationCudaCandidate::parameter_digest(
  self : Self,
) -> @joint_diffusion_vae_reference.JointVaeAudioDecoderAmpActivationDigest {
  self.parameter_digest
}

///|
pub fn JointDiffusionAudioVaeAmpActivationCudaCandidate::request_shape(
  self : Self,
) -> @joint_diffusion_plan.JointDiffusionRequestShape {
  self.request_shape
}

///|
pub fn JointDiffusionAudioVaeAmpActivationCudaCandidate::stage_index(
  self : Self,
) -> Int {
  self.stage_index
}

///|
pub fn JointDiffusionAudioVaeAmpActivationCudaCandidate::batch_count(
  self : Self,
) -> Int {
  self.batch_count
}

///|
pub fn JointDiffusionAudioVaeAmpActivationCudaCandidate::channel_count(
  self : Self,
) -> Int {
  self.channel_count
}

///|
pub fn JointDiffusionAudioVaeAmpActivationCudaCandidate::time_count(
  self : Self,
) -> Int {
  self.time_count
}

///|
pub fn JointDiffusionAudioVaeAmpActivationCudaCandidate::upsample_time(
  self : Self,
) -> Int {
  self.upsample_time
}

///|
pub fn JointDiffusionAudioVaeAmpActivationCudaCandidate::element_count(
  self : Self,
) -> Int64 {
  self.element_count
}

///|
pub fn JointDiffusionAudioVaeAmpActivationCudaCandidate::intermediate_element_count(
  self : Self,
) -> Int64 {
  self.intermediate_element_count
}

///|
pub fn JointDiffusionAudioVaeAmpActivationCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn JointDiffusionAudioVaeAmpActivationCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[
  @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
] {
  self.operands
}

///|
pub fn JointDiffusionAudioVaeAmpActivationCudaCandidate::stages(
  self : Self,
) -> ReadOnlyArray[JointDiffusionAudioVaeFirstAmpCudaStage] {
  self.stages
}

///|
pub fn JointDiffusionAudioVaeAmpActivationCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn JointDiffusionAudioVaeAmpActivationCudaCandidate::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

///|
pub fn JointDiffusionAudioVaeAmpActivationCudaCandidate::source_digest(
  self : Self,
) -> JointDiffusionCudaSourceDigest {
  self.source_digest
}

///|
pub fn JointDiffusionAudioVaeAmpActivationCudaCandidate::recipe_bytes(
  self : Self,
) -> Bytes {
  self.recipe
}

///|
pub fn JointDiffusionAudioVaeAmpActivationCudaCandidate::recipe_digest(
  self : Self,
) -> JointDiffusionCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn JointDiffusionAudioVaeAmpActivationCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

///|
pub fn JointDiffusionAudioVaeAmpStageCudaCandidate::numerical_contract(
  self : Self,
) -> JointDiffusionCudaNumericalContract {
  self.numerical_contract
}

///|
pub fn JointDiffusionAudioVaeAmpStageCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn JointDiffusionAudioVaeAmpStageCudaCandidate::requirements_digest(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn JointDiffusionAudioVaeAmpStageCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn JointDiffusionAudioVaeAmpStageCudaCandidate::operation(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation {
  self.operation
}

///|
pub fn JointDiffusionAudioVaeAmpStageCudaCandidate::capability(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionAotCapability {
  self.capability
}

///|
pub fn JointDiffusionAudioVaeAmpStageCudaCandidate::request_shape(
  self : Self,
) -> @joint_diffusion_plan.JointDiffusionRequestShape {
  self.request_shape
}

///|
pub fn JointDiffusionAudioVaeAmpStageCudaCandidate::stage_index(
  self : Self,
) -> Int {
  self.stage_index
}

///|
pub fn JointDiffusionAudioVaeAmpStageCudaCandidate::parameter_layout_digest(
  self : Self,
) -> @joint_audio_amp_parameter_layout.JointAudioAmpParameterLayoutDigest {
  self.parameter_layout_digest
}

///|
pub fn JointDiffusionAudioVaeAmpStageCudaCandidate::workspace_layout_digest(
  self : Self,
) -> @joint_audio_amp_workspace_layout.JointAudioAmpWorkspaceLayoutDigest {
  self.workspace_layout_digest
}

///|
pub fn JointDiffusionAudioVaeAmpStageCudaCandidate::batch_count(
  self : Self,
) -> Int {
  self.batch_count
}

///|
pub fn JointDiffusionAudioVaeAmpStageCudaCandidate::channel_count(
  self : Self,
) -> Int {
  self.channel_count
}

///|
pub fn JointDiffusionAudioVaeAmpStageCudaCandidate::time_count(
  self : Self,
) -> Int {
  self.time_count
}

///|
pub fn JointDiffusionAudioVaeAmpStageCudaCandidate::element_count(
  self : Self,
) -> Int64 {
  self.element_count
}

///|
pub fn JointDiffusionAudioVaeAmpStageCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn JointDiffusionAudioVaeAmpStageCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[
  @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
] {
  self.operands
}

///|
pub fn JointDiffusionAudioVaeAmpStageCudaCandidate::stages(
  self : Self,
) -> ReadOnlyArray[JointDiffusionAudioVaeFirstAmpExecutionStage] {
  self.stages
}

///|
pub fn JointDiffusionAudioVaeAmpStageCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn JointDiffusionAudioVaeAmpStageCudaCandidate::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

///|
pub fn JointDiffusionAudioVaeAmpStageCudaCandidate::source_digest(
  self : Self,
) -> JointDiffusionCudaSourceDigest {
  self.source_digest
}

///|
pub fn JointDiffusionAudioVaeAmpStageCudaCandidate::recipe_bytes(
  self : Self,
) -> Bytes {
  self.recipe
}

///|
pub fn JointDiffusionAudioVaeAmpStageCudaCandidate::recipe_digest(
  self : Self,
) -> JointDiffusionCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn JointDiffusionAudioVaeAmpStageCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

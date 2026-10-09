///|
pub fn JointDiffusionAudioVaeFirstAmpCudaStage::kind(
  self : Self,
) -> JointDiffusionAudioVaeFirstAmpStageKind {
  self.kind
}

///|
pub fn JointDiffusionAudioVaeFirstAmpCudaStage::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn JointDiffusionAudioVaeFirstAmpCudaStage::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn JointDiffusionAudioVaeFirstAmpCudaStage::operands(
  self : Self,
) -> ReadOnlyArray[
  @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
] {
  self.operands
}

///|
pub fn JointDiffusionAudioVaeFirstAmpCudaCandidate::numerical_contract(
  self : Self,
) -> JointDiffusionCudaNumericalContract {
  self.numerical_contract
}

///|
pub fn JointDiffusionAudioVaeFirstAmpCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn JointDiffusionAudioVaeFirstAmpCudaCandidate::requirements_digest(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn JointDiffusionAudioVaeFirstAmpCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn JointDiffusionAudioVaeFirstAmpCudaCandidate::operation(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation {
  self.operation
}

///|
pub fn JointDiffusionAudioVaeFirstAmpCudaCandidate::capability(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionAotCapability {
  self.capability
}

///|
pub fn JointDiffusionAudioVaeFirstAmpCudaCandidate::parameter_digest(
  self : Self,
) -> @joint_diffusion_vae_reference.JointVaeAudioDecoderFirstAmpActivationDigest {
  self.parameter_digest
}

///|
pub fn JointDiffusionAudioVaeFirstAmpCudaCandidate::request_shape(
  self : Self,
) -> @joint_diffusion_plan.JointDiffusionRequestShape {
  self.request_shape
}

///|
pub fn JointDiffusionAudioVaeFirstAmpCudaCandidate::batch_count(
  self : Self,
) -> Int {
  self.batch_count
}

///|
pub fn JointDiffusionAudioVaeFirstAmpCudaCandidate::channel_count(
  self : Self,
) -> Int {
  self.channel_count
}

///|
pub fn JointDiffusionAudioVaeFirstAmpCudaCandidate::time_count(
  self : Self,
) -> Int {
  self.time_count
}

///|
pub fn JointDiffusionAudioVaeFirstAmpCudaCandidate::upsample_time(
  self : Self,
) -> Int {
  self.upsample_time
}

///|
pub fn JointDiffusionAudioVaeFirstAmpCudaCandidate::element_count(
  self : Self,
) -> Int64 {
  self.element_count
}

///|
pub fn JointDiffusionAudioVaeFirstAmpCudaCandidate::intermediate_element_count(
  self : Self,
) -> Int64 {
  self.intermediate_element_count
}

///|
pub fn JointDiffusionAudioVaeFirstAmpCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn JointDiffusionAudioVaeFirstAmpCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[
  @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
] {
  self.operands
}

///|
pub fn JointDiffusionAudioVaeFirstAmpCudaCandidate::stages(
  self : Self,
) -> ReadOnlyArray[JointDiffusionAudioVaeFirstAmpCudaStage] {
  self.stages
}

///|
pub fn JointDiffusionAudioVaeFirstAmpCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn JointDiffusionAudioVaeFirstAmpCudaCandidate::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

///|
pub fn JointDiffusionAudioVaeFirstAmpCudaCandidate::source_digest(
  self : Self,
) -> JointDiffusionCudaSourceDigest {
  self.source_digest
}

///|
pub fn JointDiffusionAudioVaeFirstAmpCudaCandidate::recipe_bytes(
  self : Self,
) -> Bytes {
  self.recipe
}

///|
pub fn JointDiffusionAudioVaeFirstAmpCudaCandidate::recipe_digest(
  self : Self,
) -> JointDiffusionCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn JointDiffusionAudioVaeFirstAmpCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

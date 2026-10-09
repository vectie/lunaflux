///|
pub fn JointDiffusionAudioVaeLatentDestandardizationCudaCandidate::numerical_contract(
  self : Self,
) -> JointDiffusionCudaNumericalContract {
  self.numerical_contract
}

///|
pub fn JointDiffusionAudioVaeLatentDestandardizationCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn JointDiffusionAudioVaeLatentDestandardizationCudaCandidate::requirements_digest(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn JointDiffusionAudioVaeLatentDestandardizationCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn JointDiffusionAudioVaeLatentDestandardizationCudaCandidate::operation(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation {
  self.operation
}

///|
pub fn JointDiffusionAudioVaeLatentDestandardizationCudaCandidate::capability(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionAotCapability {
  self.capability
}

///|
pub fn JointDiffusionAudioVaeLatentDestandardizationCudaCandidate::parameter_digest(
  self : Self,
) -> @joint_diffusion_vae_reference.JointVaeLatentDestandardizationDigest {
  self.parameter_digest
}

///|
pub fn JointDiffusionAudioVaeLatentDestandardizationCudaCandidate::request_shape(
  self : Self,
) -> @joint_diffusion_plan.JointDiffusionRequestShape {
  self.request_shape
}

///|
pub fn JointDiffusionAudioVaeLatentDestandardizationCudaCandidate::decoder_batch_count(
  self : Self,
) -> Int {
  self.decoder_batch_count
}

///|
pub fn JointDiffusionAudioVaeLatentDestandardizationCudaCandidate::channel_count(
  self : Self,
) -> Int {
  self.channel_count
}

///|
pub fn JointDiffusionAudioVaeLatentDestandardizationCudaCandidate::elements_per_channel(
  self : Self,
) -> Int64 {
  self.elements_per_channel
}

///|
pub fn JointDiffusionAudioVaeLatentDestandardizationCudaCandidate::element_count(
  self : Self,
) -> Int64 {
  self.element_count
}

///|
pub fn JointDiffusionAudioVaeLatentDestandardizationCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn JointDiffusionAudioVaeLatentDestandardizationCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn JointDiffusionAudioVaeLatentDestandardizationCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[
  @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
] {
  self.operands
}

///|
pub fn JointDiffusionAudioVaeLatentDestandardizationCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn JointDiffusionAudioVaeLatentDestandardizationCudaCandidate::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn JointDiffusionAudioVaeLatentDestandardizationCudaCandidate::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

///|
pub fn JointDiffusionAudioVaeLatentDestandardizationCudaCandidate::source_digest(
  self : Self,
) -> JointDiffusionCudaSourceDigest {
  self.source_digest
}

///|
pub fn JointDiffusionAudioVaeLatentDestandardizationCudaCandidate::recipe_bytes(
  self : Self,
) -> Bytes {
  self.recipe
}

///|
pub fn JointDiffusionAudioVaeLatentDestandardizationCudaCandidate::recipe_digest(
  self : Self,
) -> JointDiffusionCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn JointDiffusionAudioVaeLatentDestandardizationCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

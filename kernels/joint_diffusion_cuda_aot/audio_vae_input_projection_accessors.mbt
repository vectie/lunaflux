///|
pub fn JointDiffusionAudioVaeDecoderInputProjectionCudaCandidate::numerical_contract(
  self : Self,
) -> JointDiffusionCudaNumericalContract {
  self.numerical_contract
}

///|
pub fn JointDiffusionAudioVaeDecoderInputProjectionCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn JointDiffusionAudioVaeDecoderInputProjectionCudaCandidate::requirements_digest(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn JointDiffusionAudioVaeDecoderInputProjectionCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn JointDiffusionAudioVaeDecoderInputProjectionCudaCandidate::operation(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation {
  self.operation
}

///|
pub fn JointDiffusionAudioVaeDecoderInputProjectionCudaCandidate::capability(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionAotCapability {
  self.capability
}

///|
pub fn JointDiffusionAudioVaeDecoderInputProjectionCudaCandidate::parameter_digest(
  self : Self,
) -> @joint_diffusion_vae_reference.JointVaeAudioDecoderInputProjectionDigest {
  self.parameter_digest
}

///|
pub fn JointDiffusionAudioVaeDecoderInputProjectionCudaCandidate::request_shape(
  self : Self,
) -> @joint_diffusion_plan.JointDiffusionRequestShape {
  self.request_shape
}

///|
pub fn JointDiffusionAudioVaeDecoderInputProjectionCudaCandidate::decoder_batch_count(
  self : Self,
) -> Int {
  self.decoder_batch_count
}

///|
pub fn JointDiffusionAudioVaeDecoderInputProjectionCudaCandidate::input_channels(
  self : Self,
) -> Int {
  self.input_channels
}

///|
pub fn JointDiffusionAudioVaeDecoderInputProjectionCudaCandidate::output_channels(
  self : Self,
) -> Int {
  self.output_channels
}

///|
pub fn JointDiffusionAudioVaeDecoderInputProjectionCudaCandidate::time_count(
  self : Self,
) -> Int {
  self.time_count
}

///|
pub fn JointDiffusionAudioVaeDecoderInputProjectionCudaCandidate::input_element_count(
  self : Self,
) -> Int64 {
  self.input_element_count
}

///|
pub fn JointDiffusionAudioVaeDecoderInputProjectionCudaCandidate::weight_element_count(
  self : Self,
) -> Int64 {
  self.weight_element_count
}

///|
pub fn JointDiffusionAudioVaeDecoderInputProjectionCudaCandidate::bias_element_count(
  self : Self,
) -> Int64 {
  self.bias_element_count
}

///|
pub fn JointDiffusionAudioVaeDecoderInputProjectionCudaCandidate::output_element_count(
  self : Self,
) -> Int64 {
  self.output_element_count
}

///|
pub fn JointDiffusionAudioVaeDecoderInputProjectionCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn JointDiffusionAudioVaeDecoderInputProjectionCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn JointDiffusionAudioVaeDecoderInputProjectionCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[
  @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
] {
  self.operands
}

///|
pub fn JointDiffusionAudioVaeDecoderInputProjectionCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn JointDiffusionAudioVaeDecoderInputProjectionCudaCandidate::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn JointDiffusionAudioVaeDecoderInputProjectionCudaCandidate::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

///|
pub fn JointDiffusionAudioVaeDecoderInputProjectionCudaCandidate::source_digest(
  self : Self,
) -> JointDiffusionCudaSourceDigest {
  self.source_digest
}

///|
pub fn JointDiffusionAudioVaeDecoderInputProjectionCudaCandidate::recipe_bytes(
  self : Self,
) -> Bytes {
  self.recipe
}

///|
pub fn JointDiffusionAudioVaeDecoderInputProjectionCudaCandidate::recipe_digest(
  self : Self,
) -> JointDiffusionCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn JointDiffusionAudioVaeDecoderInputProjectionCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

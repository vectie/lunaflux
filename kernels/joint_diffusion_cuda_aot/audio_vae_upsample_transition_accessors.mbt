///|
pub fn JointDiffusionAudioVaeUpsampleTransitionCudaCandidate::numerical_contract(
  self : Self,
) -> JointDiffusionCudaNumericalContract {
  self.numerical_contract
}

///|
pub fn JointDiffusionAudioVaeUpsampleTransitionCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn JointDiffusionAudioVaeUpsampleTransitionCudaCandidate::requirements_digest(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn JointDiffusionAudioVaeUpsampleTransitionCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn JointDiffusionAudioVaeUpsampleTransitionCudaCandidate::operation(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation {
  self.operation
}

///|
pub fn JointDiffusionAudioVaeUpsampleTransitionCudaCandidate::capability(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionAotCapability {
  self.capability
}

///|
pub fn JointDiffusionAudioVaeUpsampleTransitionCudaCandidate::parameter_digest(
  self : Self,
) -> @joint_diffusion_vae_reference.JointVaeAudioDecoderUpsampleTransitionDigest {
  self.parameter_digest
}

///|
pub fn JointDiffusionAudioVaeUpsampleTransitionCudaCandidate::request_shape(
  self : Self,
) -> @joint_diffusion_plan.JointDiffusionRequestShape {
  self.request_shape
}

///|
pub fn JointDiffusionAudioVaeUpsampleTransitionCudaCandidate::stage_index(
  self : Self,
) -> Int {
  self.stage_index
}

///|
pub fn JointDiffusionAudioVaeUpsampleTransitionCudaCandidate::decoder_batch_count(
  self : Self,
) -> Int {
  self.decoder_batch_count
}

///|
pub fn JointDiffusionAudioVaeUpsampleTransitionCudaCandidate::input_channels(
  self : Self,
) -> Int {
  self.input_channels
}

///|
pub fn JointDiffusionAudioVaeUpsampleTransitionCudaCandidate::output_channels(
  self : Self,
) -> Int {
  self.output_channels
}

///|
pub fn JointDiffusionAudioVaeUpsampleTransitionCudaCandidate::input_time(
  self : Self,
) -> Int {
  self.input_time
}

///|
pub fn JointDiffusionAudioVaeUpsampleTransitionCudaCandidate::output_time(
  self : Self,
) -> Int {
  self.output_time
}

///|
pub fn JointDiffusionAudioVaeUpsampleTransitionCudaCandidate::kernel_size(
  self : Self,
) -> Int {
  self.kernel_size
}

///|
pub fn JointDiffusionAudioVaeUpsampleTransitionCudaCandidate::stride(
  self : Self,
) -> Int {
  self.stride
}

///|
pub fn JointDiffusionAudioVaeUpsampleTransitionCudaCandidate::padding(
  self : Self,
) -> Int {
  self.padding
}

///|
pub fn JointDiffusionAudioVaeUpsampleTransitionCudaCandidate::input_element_count(
  self : Self,
) -> Int64 {
  self.input_element_count
}

///|
pub fn JointDiffusionAudioVaeUpsampleTransitionCudaCandidate::weight_g_element_count(
  self : Self,
) -> Int64 {
  self.weight_g_element_count
}

///|
pub fn JointDiffusionAudioVaeUpsampleTransitionCudaCandidate::weight_v_element_count(
  self : Self,
) -> Int64 {
  self.weight_v_element_count
}

///|
pub fn JointDiffusionAudioVaeUpsampleTransitionCudaCandidate::bias_element_count(
  self : Self,
) -> Int64 {
  self.bias_element_count
}

///|
pub fn JointDiffusionAudioVaeUpsampleTransitionCudaCandidate::output_element_count(
  self : Self,
) -> Int64 {
  self.output_element_count
}

///|
pub fn JointDiffusionAudioVaeUpsampleTransitionCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn JointDiffusionAudioVaeUpsampleTransitionCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[
  @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
] {
  self.operands
}

///|
pub fn JointDiffusionAudioVaeUpsampleTransitionCudaCandidate::stages(
  self : Self,
) -> ReadOnlyArray[JointDiffusionAudioVaeFirstUpsampleCudaStage] {
  self.stages
}

///|
pub fn JointDiffusionAudioVaeUpsampleTransitionCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn JointDiffusionAudioVaeUpsampleTransitionCudaCandidate::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

///|
pub fn JointDiffusionAudioVaeUpsampleTransitionCudaCandidate::source_digest(
  self : Self,
) -> JointDiffusionCudaSourceDigest {
  self.source_digest
}

///|
pub fn JointDiffusionAudioVaeUpsampleTransitionCudaCandidate::recipe_bytes(
  self : Self,
) -> Bytes {
  self.recipe
}

///|
pub fn JointDiffusionAudioVaeUpsampleTransitionCudaCandidate::recipe_digest(
  self : Self,
) -> JointDiffusionCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn JointDiffusionAudioVaeUpsampleTransitionCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

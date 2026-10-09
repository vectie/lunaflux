///|
pub fn JointDiffusionAudioVaeFirstUpsampleCudaStage::kind(
  self : Self,
) -> JointDiffusionAudioVaeFirstUpsampleStageKind {
  self.kind
}

///|
pub fn JointDiffusionAudioVaeFirstUpsampleCudaStage::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn JointDiffusionAudioVaeFirstUpsampleCudaStage::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn JointDiffusionAudioVaeFirstUpsampleCudaStage::operands(
  self : Self,
) -> ReadOnlyArray[
  @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
] {
  self.operands
}

///|
pub fn JointDiffusionAudioVaeFirstUpsampleCudaCandidate::numerical_contract(
  self : Self,
) -> JointDiffusionCudaNumericalContract {
  self.numerical_contract
}

///|
pub fn JointDiffusionAudioVaeFirstUpsampleCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn JointDiffusionAudioVaeFirstUpsampleCudaCandidate::requirements_digest(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn JointDiffusionAudioVaeFirstUpsampleCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn JointDiffusionAudioVaeFirstUpsampleCudaCandidate::operation(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation {
  self.operation
}

///|
pub fn JointDiffusionAudioVaeFirstUpsampleCudaCandidate::capability(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionAotCapability {
  self.capability
}

///|
pub fn JointDiffusionAudioVaeFirstUpsampleCudaCandidate::parameter_digest(
  self : Self,
) -> @joint_diffusion_vae_reference.JointVaeAudioDecoderFirstUpsampleDigest {
  self.parameter_digest
}

///|
pub fn JointDiffusionAudioVaeFirstUpsampleCudaCandidate::request_shape(
  self : Self,
) -> @joint_diffusion_plan.JointDiffusionRequestShape {
  self.request_shape
}

///|
pub fn JointDiffusionAudioVaeFirstUpsampleCudaCandidate::decoder_batch_count(
  self : Self,
) -> Int {
  self.decoder_batch_count
}

///|
pub fn JointDiffusionAudioVaeFirstUpsampleCudaCandidate::input_channels(
  self : Self,
) -> Int {
  self.input_channels
}

///|
pub fn JointDiffusionAudioVaeFirstUpsampleCudaCandidate::output_channels(
  self : Self,
) -> Int {
  self.output_channels
}

///|
pub fn JointDiffusionAudioVaeFirstUpsampleCudaCandidate::input_time(
  self : Self,
) -> Int {
  self.input_time
}

///|
pub fn JointDiffusionAudioVaeFirstUpsampleCudaCandidate::output_time(
  self : Self,
) -> Int {
  self.output_time
}

///|
pub fn JointDiffusionAudioVaeFirstUpsampleCudaCandidate::kernel_size(
  self : Self,
) -> Int {
  self.kernel_size
}

///|
pub fn JointDiffusionAudioVaeFirstUpsampleCudaCandidate::stride(
  self : Self,
) -> Int {
  self.stride
}

///|
pub fn JointDiffusionAudioVaeFirstUpsampleCudaCandidate::padding(
  self : Self,
) -> Int {
  self.padding
}

///|
pub fn JointDiffusionAudioVaeFirstUpsampleCudaCandidate::input_element_count(
  self : Self,
) -> Int64 {
  self.input_element_count
}

///|
pub fn JointDiffusionAudioVaeFirstUpsampleCudaCandidate::weight_g_element_count(
  self : Self,
) -> Int64 {
  self.weight_g_element_count
}

///|
pub fn JointDiffusionAudioVaeFirstUpsampleCudaCandidate::weight_v_element_count(
  self : Self,
) -> Int64 {
  self.weight_v_element_count
}

///|
pub fn JointDiffusionAudioVaeFirstUpsampleCudaCandidate::bias_element_count(
  self : Self,
) -> Int64 {
  self.bias_element_count
}

///|
pub fn JointDiffusionAudioVaeFirstUpsampleCudaCandidate::output_element_count(
  self : Self,
) -> Int64 {
  self.output_element_count
}

///|
pub fn JointDiffusionAudioVaeFirstUpsampleCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn JointDiffusionAudioVaeFirstUpsampleCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[
  @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
] {
  self.operands
}

///|
pub fn JointDiffusionAudioVaeFirstUpsampleCudaCandidate::stages(
  self : Self,
) -> ReadOnlyArray[JointDiffusionAudioVaeFirstUpsampleCudaStage] {
  self.stages
}

///|
pub fn JointDiffusionAudioVaeFirstUpsampleCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn JointDiffusionAudioVaeFirstUpsampleCudaCandidate::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

///|
pub fn JointDiffusionAudioVaeFirstUpsampleCudaCandidate::source_digest(
  self : Self,
) -> JointDiffusionCudaSourceDigest {
  self.source_digest
}

///|
pub fn JointDiffusionAudioVaeFirstUpsampleCudaCandidate::recipe_bytes(
  self : Self,
) -> Bytes {
  self.recipe
}

///|
pub fn JointDiffusionAudioVaeFirstUpsampleCudaCandidate::recipe_digest(
  self : Self,
) -> JointDiffusionCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn JointDiffusionAudioVaeFirstUpsampleCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

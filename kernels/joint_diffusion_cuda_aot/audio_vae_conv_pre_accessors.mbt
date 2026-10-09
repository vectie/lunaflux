///|
pub fn JointDiffusionAudioVaeConvPreCudaStage::kind(
  self : Self,
) -> JointDiffusionAudioVaeConvPreStageKind {
  self.kind
}

///|
pub fn JointDiffusionAudioVaeConvPreCudaStage::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn JointDiffusionAudioVaeConvPreCudaStage::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn JointDiffusionAudioVaeConvPreCudaStage::operands(
  self : Self,
) -> ReadOnlyArray[
  @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
] {
  self.operands
}

///|
pub fn JointDiffusionAudioVaeConvPreCudaCandidate::numerical_contract(
  self : Self,
) -> JointDiffusionCudaNumericalContract {
  self.numerical_contract
}

///|
pub fn JointDiffusionAudioVaeConvPreCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn JointDiffusionAudioVaeConvPreCudaCandidate::requirements_digest(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn JointDiffusionAudioVaeConvPreCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn JointDiffusionAudioVaeConvPreCudaCandidate::operation(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation {
  self.operation
}

///|
pub fn JointDiffusionAudioVaeConvPreCudaCandidate::capability(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionAotCapability {
  self.capability
}

///|
pub fn JointDiffusionAudioVaeConvPreCudaCandidate::parameter_digest(
  self : Self,
) -> @joint_diffusion_vae_reference.JointVaeAudioDecoderConvPreDigest {
  self.parameter_digest
}

///|
pub fn JointDiffusionAudioVaeConvPreCudaCandidate::request_shape(
  self : Self,
) -> @joint_diffusion_plan.JointDiffusionRequestShape {
  self.request_shape
}

///|
pub fn JointDiffusionAudioVaeConvPreCudaCandidate::decoder_batch_count(
  self : Self,
) -> Int {
  self.decoder_batch_count
}

///|
pub fn JointDiffusionAudioVaeConvPreCudaCandidate::input_channels(
  self : Self,
) -> Int {
  self.input_channels
}

///|
pub fn JointDiffusionAudioVaeConvPreCudaCandidate::output_channels(
  self : Self,
) -> Int {
  self.output_channels
}

///|
pub fn JointDiffusionAudioVaeConvPreCudaCandidate::time_count(
  self : Self,
) -> Int {
  self.time_count
}

///|
pub fn JointDiffusionAudioVaeConvPreCudaCandidate::kernel_size(
  self : Self,
) -> Int {
  self.kernel_size
}

///|
pub fn JointDiffusionAudioVaeConvPreCudaCandidate::padding(self : Self) -> Int {
  self.padding
}

///|
pub fn JointDiffusionAudioVaeConvPreCudaCandidate::stride(self : Self) -> Int {
  self.stride
}

///|
pub fn JointDiffusionAudioVaeConvPreCudaCandidate::input_element_count(
  self : Self,
) -> Int64 {
  self.input_element_count
}

///|
pub fn JointDiffusionAudioVaeConvPreCudaCandidate::weight_g_element_count(
  self : Self,
) -> Int64 {
  self.weight_g_element_count
}

///|
pub fn JointDiffusionAudioVaeConvPreCudaCandidate::weight_v_element_count(
  self : Self,
) -> Int64 {
  self.weight_v_element_count
}

///|
pub fn JointDiffusionAudioVaeConvPreCudaCandidate::bias_element_count(
  self : Self,
) -> Int64 {
  self.bias_element_count
}

///|
pub fn JointDiffusionAudioVaeConvPreCudaCandidate::output_element_count(
  self : Self,
) -> Int64 {
  self.output_element_count
}

///|
pub fn JointDiffusionAudioVaeConvPreCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn JointDiffusionAudioVaeConvPreCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[
  @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
] {
  self.operands
}

///|
pub fn JointDiffusionAudioVaeConvPreCudaCandidate::stages(
  self : Self,
) -> ReadOnlyArray[JointDiffusionAudioVaeConvPreCudaStage] {
  self.stages
}

///|
pub fn JointDiffusionAudioVaeConvPreCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn JointDiffusionAudioVaeConvPreCudaCandidate::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

///|
pub fn JointDiffusionAudioVaeConvPreCudaCandidate::source_digest(
  self : Self,
) -> JointDiffusionCudaSourceDigest {
  self.source_digest
}

///|
pub fn JointDiffusionAudioVaeConvPreCudaCandidate::recipe_bytes(
  self : Self,
) -> Bytes {
  self.recipe
}

///|
pub fn JointDiffusionAudioVaeConvPreCudaCandidate::recipe_digest(
  self : Self,
) -> JointDiffusionCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn JointDiffusionAudioVaeConvPreCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}

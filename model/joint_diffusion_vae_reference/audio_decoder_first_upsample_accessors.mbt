///|
pub fn JointVaeAudioDecoderFirstUpsample::plan_digest(
  self : Self,
) -> @spec.PlanDigest {
  self.plan_digest
}

///|
pub fn JointVaeAudioDecoderFirstUpsample::request_shape(
  self : Self,
) -> @joint_diffusion_plan.JointDiffusionRequestShape {
  self.request_shape
}

///|
pub fn JointVaeAudioDecoderFirstUpsample::decoder_batch_count(
  self : Self,
) -> Int {
  self.decoder_batch_count
}

///|
pub fn JointVaeAudioDecoderFirstUpsample::input_channels(self : Self) -> Int {
  self.input_channels
}

///|
pub fn JointVaeAudioDecoderFirstUpsample::output_channels(self : Self) -> Int {
  self.output_channels
}

///|
pub fn JointVaeAudioDecoderFirstUpsample::input_time(self : Self) -> Int {
  self.input_time
}

///|
pub fn JointVaeAudioDecoderFirstUpsample::output_time(self : Self) -> Int {
  self.output_time
}

///|
pub fn JointVaeAudioDecoderFirstUpsample::kernel_size(self : Self) -> Int {
  self.kernel_size
}

///|
pub fn JointVaeAudioDecoderFirstUpsample::stride(self : Self) -> Int {
  self.stride
}

///|
pub fn JointVaeAudioDecoderFirstUpsample::padding(self : Self) -> Int {
  self.padding
}

///|
pub fn JointVaeAudioDecoderFirstUpsample::storage(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionAudioVaeConvTransposeStorage {
  self.storage
}

///|
pub fn JointVaeAudioDecoderFirstUpsample::axes(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionAudioVaeWeightNormAxes {
  self.axes
}

///|
pub fn JointVaeAudioDecoderFirstUpsample::layout(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionAudioVaeLatentLayout {
  self.layout
}

///|
pub fn JointVaeAudioDecoderFirstUpsample::weight_g_name(self : Self) -> String {
  self.weight_g_name
}

///|
pub fn JointVaeAudioDecoderFirstUpsample::weight_v_name(self : Self) -> String {
  self.weight_v_name
}

///|
pub fn JointVaeAudioDecoderFirstUpsample::bias_name(self : Self) -> String {
  self.bias_name
}

///|
pub fn JointVaeAudioDecoderFirstUpsample::input_element_count(
  self : Self,
) -> Int64 {
  self.input_element_count
}

///|
pub fn JointVaeAudioDecoderFirstUpsample::weight_g_element_count(
  self : Self,
) -> Int64 {
  self.weight_g_element_count
}

///|
pub fn JointVaeAudioDecoderFirstUpsample::weight_v_element_count(
  self : Self,
) -> Int64 {
  self.weight_v_element_count
}

///|
pub fn JointVaeAudioDecoderFirstUpsample::bias_element_count(
  self : Self,
) -> Int64 {
  self.bias_element_count
}

///|
pub fn JointVaeAudioDecoderFirstUpsample::output_element_count(
  self : Self,
) -> Int64 {
  self.output_element_count
}

///|
pub fn JointVaeAudioDecoderFirstUpsample::digest(
  self : Self,
) -> JointVaeAudioDecoderFirstUpsampleDigest {
  self.digest
}

///|
pub fn JointVaeAudioDecoderFirstUpsampleDigest::as_hex(self : Self) -> String {
  self.hex
}

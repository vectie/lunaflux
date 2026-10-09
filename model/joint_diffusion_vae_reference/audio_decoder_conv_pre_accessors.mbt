///|
pub fn JointVaeAudioDecoderConvPre::plan_digest(
  self : Self,
) -> @spec.PlanDigest {
  self.plan_digest
}

///|
pub fn JointVaeAudioDecoderConvPre::request_shape(
  self : Self,
) -> @joint_diffusion_plan.JointDiffusionRequestShape {
  self.request_shape
}

///|
pub fn JointVaeAudioDecoderConvPre::decoder_batch_count(self : Self) -> Int {
  self.decoder_batch_count
}

///|
pub fn JointVaeAudioDecoderConvPre::input_channels(self : Self) -> Int {
  self.input_channels
}

///|
pub fn JointVaeAudioDecoderConvPre::output_channels(self : Self) -> Int {
  self.output_channels
}

///|
pub fn JointVaeAudioDecoderConvPre::kernel_size(self : Self) -> Int {
  self.kernel_size
}

///|
pub fn JointVaeAudioDecoderConvPre::padding(self : Self) -> Int {
  self.padding
}

///|
pub fn JointVaeAudioDecoderConvPre::stride(self : Self) -> Int {
  self.stride
}

///|
pub fn JointVaeAudioDecoderConvPre::storage(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionAudioVaeConvPreStorage {
  self.storage
}

///|
pub fn JointVaeAudioDecoderConvPre::axes(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionAudioVaeWeightNormAxes {
  self.axes
}

///|
pub fn JointVaeAudioDecoderConvPre::layout(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionAudioVaeLatentLayout {
  self.layout
}

///|
pub fn JointVaeAudioDecoderConvPre::weight_g_name(self : Self) -> String {
  self.weight_g_name
}

///|
pub fn JointVaeAudioDecoderConvPre::weight_v_name(self : Self) -> String {
  self.weight_v_name
}

///|
pub fn JointVaeAudioDecoderConvPre::bias_name(self : Self) -> String {
  self.bias_name
}

///|
pub fn JointVaeAudioDecoderConvPre::input_element_count(self : Self) -> Int64 {
  self.input_element_count
}

///|
pub fn JointVaeAudioDecoderConvPre::weight_g_element_count(
  self : Self,
) -> Int64 {
  self.weight_g_element_count
}

///|
pub fn JointVaeAudioDecoderConvPre::weight_v_element_count(
  self : Self,
) -> Int64 {
  self.weight_v_element_count
}

///|
pub fn JointVaeAudioDecoderConvPre::bias_element_count(self : Self) -> Int64 {
  self.bias_element_count
}

///|
pub fn JointVaeAudioDecoderConvPre::output_element_count(self : Self) -> Int64 {
  self.output_element_count
}

///|
pub fn JointVaeAudioDecoderConvPre::digest(
  self : Self,
) -> JointVaeAudioDecoderConvPreDigest {
  self.digest
}

///|
pub fn JointVaeAudioDecoderConvPreDigest::as_hex(self : Self) -> String {
  self.hex
}

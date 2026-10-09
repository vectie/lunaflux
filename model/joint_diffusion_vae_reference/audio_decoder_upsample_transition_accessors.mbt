///|
pub fn JointVaeAudioDecoderUpsampleTransition::plan_digest(
  self : Self,
) -> @spec.PlanDigest {
  self.plan_digest
}

///|
pub fn JointVaeAudioDecoderUpsampleTransition::request_shape(
  self : Self,
) -> @joint_diffusion_plan.JointDiffusionRequestShape {
  self.request_shape
}

///|
pub fn JointVaeAudioDecoderUpsampleTransition::stage_index(self : Self) -> Int {
  self.stage_index
}

///|
pub fn JointVaeAudioDecoderUpsampleTransition::decoder_batch_count(
  self : Self,
) -> Int {
  self.decoder_batch_count
}

///|
pub fn JointVaeAudioDecoderUpsampleTransition::input_channels(
  self : Self,
) -> Int {
  self.input_channels
}

///|
pub fn JointVaeAudioDecoderUpsampleTransition::output_channels(
  self : Self,
) -> Int {
  self.output_channels
}

///|
pub fn JointVaeAudioDecoderUpsampleTransition::input_time(self : Self) -> Int {
  self.input_time
}

///|
pub fn JointVaeAudioDecoderUpsampleTransition::output_time(self : Self) -> Int {
  self.output_time
}

///|
pub fn JointVaeAudioDecoderUpsampleTransition::kernel_size(self : Self) -> Int {
  self.kernel_size
}

///|
pub fn JointVaeAudioDecoderUpsampleTransition::stride(self : Self) -> Int {
  self.stride
}

///|
pub fn JointVaeAudioDecoderUpsampleTransition::padding(self : Self) -> Int {
  self.padding
}

///|
pub fn JointVaeAudioDecoderUpsampleTransition::storage(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionAudioVaeConvTransposeStorage {
  self.storage
}

///|
pub fn JointVaeAudioDecoderUpsampleTransition::axes(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionAudioVaeWeightNormAxes {
  self.axes
}

///|
pub fn JointVaeAudioDecoderUpsampleTransition::layout(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionAudioVaeLatentLayout {
  self.layout
}

///|
pub fn JointVaeAudioDecoderUpsampleTransition::weight_g_name(
  self : Self,
) -> String {
  self.weight_g_name
}

///|
pub fn JointVaeAudioDecoderUpsampleTransition::weight_v_name(
  self : Self,
) -> String {
  self.weight_v_name
}

///|
pub fn JointVaeAudioDecoderUpsampleTransition::bias_name(self : Self) -> String {
  self.bias_name
}

///|
pub fn JointVaeAudioDecoderUpsampleTransition::input_element_count(
  self : Self,
) -> Int64 {
  self.input_element_count
}

///|
pub fn JointVaeAudioDecoderUpsampleTransition::weight_g_element_count(
  self : Self,
) -> Int64 {
  self.weight_g_element_count
}

///|
pub fn JointVaeAudioDecoderUpsampleTransition::weight_v_element_count(
  self : Self,
) -> Int64 {
  self.weight_v_element_count
}

///|
pub fn JointVaeAudioDecoderUpsampleTransition::bias_element_count(
  self : Self,
) -> Int64 {
  self.bias_element_count
}

///|
pub fn JointVaeAudioDecoderUpsampleTransition::output_element_count(
  self : Self,
) -> Int64 {
  self.output_element_count
}

///|
pub fn JointVaeAudioDecoderUpsampleTransition::digest(
  self : Self,
) -> JointVaeAudioDecoderUpsampleTransitionDigest {
  self.digest
}

///|
pub fn JointVaeAudioDecoderUpsampleTransitionDigest::as_hex(
  self : Self,
) -> String {
  self.hex
}

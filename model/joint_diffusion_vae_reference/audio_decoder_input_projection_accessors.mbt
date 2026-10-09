///|
pub fn JointVaeAudioDecoderInputProjection::plan_digest(
  self : Self,
) -> @spec.PlanDigest {
  self.plan_digest
}

///|
pub fn JointVaeAudioDecoderInputProjection::request_shape(
  self : Self,
) -> @joint_diffusion_plan.JointDiffusionRequestShape {
  self.request_shape
}

///|
pub fn JointVaeAudioDecoderInputProjection::decoder_batch_count(
  self : Self,
) -> Int {
  self.decoder_batch_count
}

///|
pub fn JointVaeAudioDecoderInputProjection::input_channels(self : Self) -> Int {
  self.input_channels
}

///|
pub fn JointVaeAudioDecoderInputProjection::output_channels(self : Self) -> Int {
  self.output_channels
}

///|
pub fn JointVaeAudioDecoderInputProjection::kernel_size(self : Self) -> Int {
  self.kernel_size
}

///|
pub fn JointVaeAudioDecoderInputProjection::storage(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionAudioVaeInputProjectionStorage {
  self.storage
}

///|
pub fn JointVaeAudioDecoderInputProjection::layout(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionAudioVaeLatentLayout {
  self.layout
}

///|
pub fn JointVaeAudioDecoderInputProjection::weight_name(self : Self) -> String {
  self.weight_name
}

///|
pub fn JointVaeAudioDecoderInputProjection::bias_name(self : Self) -> String {
  self.bias_name
}

///|
pub fn JointVaeAudioDecoderInputProjection::input_element_count(
  self : Self,
) -> Int64 {
  self.input_element_count
}

///|
pub fn JointVaeAudioDecoderInputProjection::weight_element_count(
  self : Self,
) -> Int64 {
  self.weight_element_count
}

///|
pub fn JointVaeAudioDecoderInputProjection::bias_element_count(
  self : Self,
) -> Int64 {
  self.bias_element_count
}

///|
pub fn JointVaeAudioDecoderInputProjection::output_element_count(
  self : Self,
) -> Int64 {
  self.output_element_count
}

///|
pub fn JointVaeAudioDecoderInputProjection::digest(
  self : Self,
) -> JointVaeAudioDecoderInputProjectionDigest {
  self.digest
}

///|
pub fn JointVaeAudioDecoderInputProjectionDigest::as_hex(self : Self) -> String {
  self.hex
}

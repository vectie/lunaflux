///|
pub fn JointVaeAudioDecoderFirstAmpActivation::plan_digest(
  self : Self,
) -> @spec.PlanDigest {
  self.plan_digest
}

///|
pub fn JointVaeAudioDecoderFirstAmpActivation::request_shape(
  self : Self,
) -> @joint_diffusion_plan.JointDiffusionRequestShape {
  self.request_shape
}

///|
pub fn JointVaeAudioDecoderFirstAmpActivation::decoder_batch_count(
  self : Self,
) -> Int {
  self.decoder_batch_count
}

///|
pub fn JointVaeAudioDecoderFirstAmpActivation::channel_count(
  self : Self,
) -> Int {
  self.channel_count
}

///|
pub fn JointVaeAudioDecoderFirstAmpActivation::time_count(self : Self) -> Int {
  self.time_count
}

///|
pub fn JointVaeAudioDecoderFirstAmpActivation::upsample_time(
  self : Self,
) -> Int {
  self.upsample_time
}

///|
pub fn JointVaeAudioDecoderFirstAmpActivation::ratio(self : Self) -> Int {
  self.ratio
}

///|
pub fn JointVaeAudioDecoderFirstAmpActivation::kernel_size(self : Self) -> Int {
  self.kernel_size
}

///|
pub fn JointVaeAudioDecoderFirstAmpActivation::storage(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionAudioVaeAliasFreeActivationStorage {
  self.storage
}

///|
pub fn JointVaeAudioDecoderFirstAmpActivation::layout(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionAudioVaeLatentLayout {
  self.layout
}

///|
pub fn JointVaeAudioDecoderFirstAmpActivation::alpha_name(
  self : Self,
) -> String {
  self.alpha_name
}

///|
pub fn JointVaeAudioDecoderFirstAmpActivation::beta_name(self : Self) -> String {
  self.beta_name
}

///|
pub fn JointVaeAudioDecoderFirstAmpActivation::upsample_filter_name(
  self : Self,
) -> String {
  self.upsample_filter_name
}

///|
pub fn JointVaeAudioDecoderFirstAmpActivation::downsample_filter_name(
  self : Self,
) -> String {
  self.downsample_filter_name
}

///|
pub fn JointVaeAudioDecoderFirstAmpActivation::element_count(
  self : Self,
) -> Int64 {
  self.element_count
}

///|
pub fn JointVaeAudioDecoderFirstAmpActivation::intermediate_element_count(
  self : Self,
) -> Int64 {
  self.intermediate_element_count
}

///|
pub fn JointVaeAudioDecoderFirstAmpActivation::digest(
  self : Self,
) -> JointVaeAudioDecoderFirstAmpActivationDigest {
  self.digest
}

///|
pub fn JointVaeAudioDecoderFirstAmpActivationDigest::as_hex(
  self : Self,
) -> String {
  self.hex
}

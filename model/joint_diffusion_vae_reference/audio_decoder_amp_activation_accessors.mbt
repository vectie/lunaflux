///|
pub fn JointVaeAudioDecoderAmpActivation::plan_digest(
  self : Self,
) -> @spec.PlanDigest {
  self.plan_digest
}

///|
pub fn JointVaeAudioDecoderAmpActivation::request_shape(
  self : Self,
) -> @joint_diffusion_plan.JointDiffusionRequestShape {
  self.request_shape
}

///|
pub fn JointVaeAudioDecoderAmpActivation::stage_index(self : Self) -> Int {
  self.stage_index
}

///|
pub fn JointVaeAudioDecoderAmpActivation::decoder_batch_count(
  self : Self,
) -> Int {
  self.decoder_batch_count
}

///|
pub fn JointVaeAudioDecoderAmpActivation::channel_count(self : Self) -> Int {
  self.channel_count
}

///|
pub fn JointVaeAudioDecoderAmpActivation::time_count(self : Self) -> Int {
  self.time_count
}

///|
pub fn JointVaeAudioDecoderAmpActivation::upsample_time(self : Self) -> Int {
  self.upsample_time
}

///|
pub fn JointVaeAudioDecoderAmpActivation::ratio(self : Self) -> Int {
  self.ratio
}

///|
pub fn JointVaeAudioDecoderAmpActivation::kernel_size(self : Self) -> Int {
  self.kernel_size
}

///|
pub fn JointVaeAudioDecoderAmpActivation::storage(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionAudioVaeAliasFreeActivationStorage {
  self.storage
}

///|
pub fn JointVaeAudioDecoderAmpActivation::layout(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionAudioVaeLatentLayout {
  self.layout
}

///|
pub fn JointVaeAudioDecoderAmpActivation::alpha_name(self : Self) -> String {
  self.alpha_name
}

///|
pub fn JointVaeAudioDecoderAmpActivation::beta_name(self : Self) -> String {
  self.beta_name
}

///|
pub fn JointVaeAudioDecoderAmpActivation::upsample_filter_name(
  self : Self,
) -> String {
  self.upsample_filter_name
}

///|
pub fn JointVaeAudioDecoderAmpActivation::downsample_filter_name(
  self : Self,
) -> String {
  self.downsample_filter_name
}

///|
pub fn JointVaeAudioDecoderAmpActivation::element_count(self : Self) -> Int64 {
  self.element_count
}

///|
pub fn JointVaeAudioDecoderAmpActivation::intermediate_element_count(
  self : Self,
) -> Int64 {
  self.intermediate_element_count
}

///|
pub fn JointVaeAudioDecoderAmpActivation::digest(
  self : Self,
) -> JointVaeAudioDecoderAmpActivationDigest {
  self.digest
}

///|
pub fn JointVaeAudioDecoderAmpActivationDigest::as_hex(self : Self) -> String {
  self.hex
}

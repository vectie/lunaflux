///|
pub fn JointVaeAudioLatentDestandardization::plan_digest(
  self : Self,
) -> @spec.PlanDigest {
  self.plan_digest
}

///|
pub fn JointVaeAudioLatentDestandardization::request_shape(
  self : Self,
) -> @joint_diffusion_plan.JointDiffusionRequestShape {
  self.request_shape
}

///|
pub fn JointVaeAudioLatentDestandardization::storage(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionVaeLatentDestandardizationStorage {
  self.storage
}

///|
pub fn JointVaeAudioLatentDestandardization::layout(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionAudioVaeLatentLayout {
  self.layout
}

///|
pub fn JointVaeAudioLatentDestandardization::batch_count(self : Self) -> Int {
  self.batch_count
}

///|
pub fn JointVaeAudioLatentDestandardization::channel_count(self : Self) -> Int {
  self.channel_count
}

///|
pub fn JointVaeAudioLatentDestandardization::elements_per_channel(
  self : Self,
) -> Int64 {
  self.elements_per_channel
}

///|
pub fn JointVaeAudioLatentDestandardization::element_count(
  self : Self,
) -> Int64 {
  self.element_count
}

///|
pub fn JointVaeAudioLatentDestandardization::mean(
  self : Self,
  channel : Int,
) -> Float? {
  self.means.get(channel)
}

///|
pub fn JointVaeAudioLatentDestandardization::standard_deviation(
  self : Self,
  channel : Int,
) -> Float? {
  self.standard_deviations.get(channel)
}

///|
pub fn JointVaeAudioLatentDestandardization::digest(
  self : Self,
) -> JointVaeLatentDestandardizationDigest {
  self.digest
}

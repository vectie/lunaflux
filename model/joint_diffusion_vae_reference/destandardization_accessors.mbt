///|
pub fn JointVaeLatentDestandardization::plan_digest(
  self : Self,
) -> @spec.PlanDigest {
  self.plan_digest
}

///|
pub fn JointVaeLatentDestandardization::request_shape(
  self : Self,
) -> @joint_diffusion_plan.JointDiffusionRequestShape {
  self.request_shape
}

///|
pub fn JointVaeLatentDestandardization::storage(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionVaeLatentDestandardizationStorage {
  self.storage
}

///|
pub fn JointVaeLatentDestandardization::layout(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionVideoVaeLatentLayout {
  self.layout
}

///|
pub fn JointVaeLatentDestandardization::channel_count(self : Self) -> Int {
  self.channel_count
}

///|
pub fn JointVaeLatentDestandardization::batch_count(self : Self) -> Int {
  self.batch_count
}

///|
pub fn JointVaeLatentDestandardization::elements_per_channel(
  self : Self,
) -> Int64 {
  self.elements_per_channel
}

///|
pub fn JointVaeLatentDestandardization::element_count(self : Self) -> Int64 {
  self.element_count
}

///|
pub fn JointVaeLatentDestandardization::mean(
  self : Self,
  channel : Int,
) -> Float? {
  self.means.get(channel)
}

///|
pub fn JointVaeLatentDestandardization::standard_deviation(
  self : Self,
  channel : Int,
) -> Float? {
  self.standard_deviations.get(channel)
}

///|
pub fn JointVaeLatentDestandardization::digest(
  self : Self,
) -> JointVaeLatentDestandardizationDigest {
  self.digest
}

///|
pub fn JointVaeLatentDestandardizationDigest::as_hex(self : Self) -> String {
  self.hex
}

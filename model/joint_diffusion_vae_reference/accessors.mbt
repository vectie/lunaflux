///|
pub fn JointVaeDecodeAssessment::geometry(
  self : Self,
) -> JointVaeDecodeGeometry {
  self.geometry
}

///|
pub fn JointVaeDecodeAssessment::gap_digest(
  self : Self,
) -> JointVaeDecodeGapDigest {
  self.gap_digest
}

///|
pub fn JointVaeDecodeGapDigest::as_hex(self : Self) -> String {
  self.hex
}

///|
pub fn JointVaeDecodeAssessment::video_gap_count(self : Self) -> Int {
  self.video_gaps.length()
}

///|
pub fn JointVaeDecodeAssessment::video_gap(
  self : Self,
  index : Int,
) -> JointVaeDecodeGap? {
  self.video_gaps.get(index)
}

///|
pub fn JointVaeDecodeAssessment::audio_gap_count(self : Self) -> Int {
  self.audio_gaps.length()
}

///|
pub fn JointVaeDecodeAssessment::audio_gap(
  self : Self,
  index : Int,
) -> JointVaeDecodeGap? {
  self.audio_gaps.get(index)
}

///|
pub fn JointVaeDecodeGeometry::plan_digest(self : Self) -> @spec.PlanDigest {
  self.plan_digest
}

///|
pub fn JointVaeDecodeGeometry::video_latent_channels(self : Self) -> Int {
  self.video_latent_channels
}

///|
pub fn JointVaeDecodeGeometry::video_latent_frames(self : Self) -> Int {
  self.video_latent_frames
}

///|
pub fn JointVaeDecodeGeometry::video_latent_height(self : Self) -> Int {
  self.video_latent_height
}

///|
pub fn JointVaeDecodeGeometry::video_latent_width(self : Self) -> Int {
  self.video_latent_width
}

///|
pub fn JointVaeDecodeGeometry::video_latent_elements(self : Self) -> Int64 {
  self.video_latent_elements
}

///|
pub fn JointVaeDecodeGeometry::video_output_frames(self : Self) -> Int {
  self.video_output_frames
}

///|
pub fn JointVaeDecodeGeometry::video_output_height(self : Self) -> Int {
  self.video_output_height
}

///|
pub fn JointVaeDecodeGeometry::video_output_width(self : Self) -> Int {
  self.video_output_width
}

///|
pub fn JointVaeDecodeGeometry::video_output_sites(self : Self) -> Int64 {
  self.video_output_sites
}

///|
pub fn JointVaeDecodeGeometry::video_spatial_compression(self : Self) -> Int {
  self.video_spatial_compression
}

///|
pub fn JointVaeDecodeGeometry::audio_latent_channels(self : Self) -> Int {
  self.audio_latent_channels
}

///|
pub fn JointVaeDecodeGeometry::audio_latent_frames(self : Self) -> Int {
  self.audio_latent_frames
}

///|
pub fn JointVaeDecodeGeometry::audio_latent_elements(self : Self) -> Int64 {
  self.audio_latent_elements
}

///|
pub fn JointVaeDecodeGeometry::audio_output_samples(self : Self) -> Int64 {
  self.audio_output_samples
}

///|
pub fn JointVaeDecodeGeometry::audio_output_channels(self : Self) -> Int {
  self.audio_output_channels
}

///|
pub fn JointVaeDecodeGeometry::audio_output_elements(self : Self) -> Int64 {
  self.audio_output_elements
}

///|
pub fn JointVaeLatentBuffer::component(
  self : Self,
) -> @joint_diffusion_plan.JointDiffusionComponent {
  self.component
}

///|
pub fn JointVaeLatentBuffer::plan_digest(self : Self) -> @spec.PlanDigest {
  self.plan_digest
}

///|
pub fn JointVaeLatentBuffer::values(self : Self) -> ReadOnlyArray[Double] {
  self.values
}

///|
pub fn JointDiffusionIoPlan::schema_version(self : Self) -> Int {
  self.schema_version
}

///|
pub fn JointDiffusionIoPlan::identity(self : Self) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn JointDiffusionIoPlan::request_digest(self : Self) -> @spec.PlanDigest {
  self.request_digest
}

///|
pub fn JointDiffusionIoPlan::workflow(
  self : Self,
) -> @joint_diffusion_plan.JointDiffusionWorkflow {
  self.workflow
}

///|
pub fn JointDiffusionIoPlan::text(self : Self) -> ValidatedTextConditioning {
  self.text
}

///|
pub fn JointDiffusionIoPlan::keyframes(
  self : Self,
) -> ReadOnlyArray[JointKeyframeConditioning] {
  self.keyframes
}

///|
pub fn JointDiffusionIoPlan::references(
  self : Self,
) -> ReadOnlyArray[JointReferenceConditioning] {
  self.references
}

///|
pub fn JointDiffusionIoPlan::shape(
  self : Self,
) -> @joint_diffusion_plan.JointDiffusionRequestShape {
  self.shape
}

///|
pub fn JointDiffusionIoPlan::video_latent_channels(self : Self) -> Int {
  self.video_latent_channels
}

///|
pub fn JointDiffusionIoPlan::video_patch_temporal(self : Self) -> Int {
  self.video_patch_temporal
}

///|
pub fn JointDiffusionIoPlan::video_patch_height(self : Self) -> Int {
  self.video_patch_height
}

///|
pub fn JointDiffusionIoPlan::video_patch_width(self : Self) -> Int {
  self.video_patch_width
}

///|
pub fn JointDiffusionIoPlan::audio_latent_channels(self : Self) -> Int {
  self.audio_latent_channels
}

///|
pub fn JointDiffusionIoPlan::audio_sample_rate(self : Self) -> Int {
  self.audio_sample_rate
}

///|
pub fn JointDiffusionIoPlan::audio_output_channels(self : Self) -> Int {
  self.audio_output_channels
}

///|
pub fn JointDiffusionIoPlan::seed(self : Self) -> Int64 {
  self.seed
}

///|
pub fn JointDiffusionIoPlan::guidance(self : Self) -> JointGuidancePolicy {
  self.guidance
}

///|
pub fn JointDiffusionIoPlan::cancellation(
  self : Self,
) -> JointCancellationPolicy {
  self.cancellation
}

///|
pub fn JointDiffusionIoPlan::stage_count(self : Self) -> Int {
  self.stage_count
}

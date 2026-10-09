///|
pub fn JointTinyDenoiseStepResult::plan_digest(self : Self) -> @spec.PlanDigest {
  self.plan_digest
}

///|
pub fn JointTinyDenoiseStepResult::step_index(self : Self) -> Int {
  self.step_index
}

///|
pub fn JointTinyDenoiseStepResult::initial_video(
  self : Self,
) -> ReadOnlyArray[Double] {
  self.initial_video
}

///|
pub fn JointTinyDenoiseStepResult::initial_audio(
  self : Self,
) -> ReadOnlyArray[Double] {
  self.initial_audio
}

///|
pub fn JointTinyDenoiseStepResult::packed_projected(
  self : Self,
) -> ReadOnlyArray[Double] {
  self.packed_projected
}

///|
pub fn JointTinyDenoiseStepResult::attended(
  self : Self,
) -> ReadOnlyArray[Double] {
  self.attended
}

///|
pub fn JointTinyDenoiseStepResult::video_prediction(
  self : Self,
) -> ReadOnlyArray[Double] {
  self.video_prediction
}

///|
pub fn JointTinyDenoiseStepResult::audio_prediction(
  self : Self,
) -> ReadOnlyArray[Double] {
  self.audio_prediction
}

///|
pub fn JointTinyDenoiseStepResult::updated_video(
  self : Self,
) -> ReadOnlyArray[Double] {
  self.updated_video
}

///|
pub fn JointTinyDenoiseStepResult::updated_audio(
  self : Self,
) -> ReadOnlyArray[Double] {
  self.updated_audio
}

///|
pub fn JointTinyDenoiseStepResult::video_sigma(self : Self) -> Double {
  self.video_sigma
}

///|
pub fn JointTinyDenoiseStepResult::video_next_sigma(self : Self) -> Double {
  self.video_next_sigma
}

///|
pub fn JointTinyDenoiseStepResult::audio_sigma(self : Self) -> Double {
  self.audio_sigma
}

///|
pub fn JointTinyDenoiseStepResult::audio_next_sigma(self : Self) -> Double {
  self.audio_next_sigma
}

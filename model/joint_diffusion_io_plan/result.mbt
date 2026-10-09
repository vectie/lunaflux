///|
pub fn JointDiffusionResultMetadata::succeeded(
  request : JointDiffusionIoPlan,
  video_frame_count~ : Int,
  audio_sample_count~ : Int64,
) -> JointDiffusionResultMetadata raise JointDiffusionIoPlanError {
  if video_frame_count != request.shape().frames() ||
    audio_sample_count != request.shape().audio_samples() {
    raise InvalidResultMetadata
  }
  {
    identity: request.identity(),
    request_digest: request.request_digest(),
    status: Succeeded,
    completed_stage_count: request.stage_count(),
    video_frame_count: Some(video_frame_count),
    audio_sample_count: Some(audio_sample_count),
  }
}

///|
pub fn JointDiffusionResultMetadata::cancelled(
  request : JointDiffusionIoPlan,
  reason~ : JointCancellationReason,
  completed_stage_count~ : Int,
) -> JointDiffusionResultMetadata raise JointDiffusionIoPlanError {
  if completed_stage_count < 0 || completed_stage_count >= request.stage_count() {
    raise InvalidCancellationProgress
  }
  {
    identity: request.identity(),
    request_digest: request.request_digest(),
    status: Cancelled(reason),
    completed_stage_count,
    video_frame_count: None,
    audio_sample_count: None,
  }
}

///|
pub fn JointDiffusionResultMetadata::identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn JointDiffusionResultMetadata::request_digest(
  self : Self,
) -> @spec.PlanDigest {
  self.request_digest
}

///|
pub fn JointDiffusionResultMetadata::status(
  self : Self,
) -> JointDiffusionResultStatus {
  self.status
}

///|
pub fn JointDiffusionResultMetadata::completed_stage_count(self : Self) -> Int {
  self.completed_stage_count
}

///|
pub fn JointDiffusionResultMetadata::video_frame_count(self : Self) -> Int? {
  self.video_frame_count
}

///|
pub fn JointDiffusionResultMetadata::audio_sample_count(self : Self) -> Int64? {
  self.audio_sample_count
}

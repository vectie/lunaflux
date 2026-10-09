///|
/// Complete prepared H3 execution owner. Module/bootstrap owners remain outside
/// this request; all conditioning/decoder stage ownership transfers as specified
/// by prepare. Keep this request after failure and close before module buffers.
pub struct Request {
  priv pipeline : @joint_diffusion_device.PipelineProgram
  priv mut video : @video.VideoDecodeRequest?
  priv mut audio : @blocks.AudioDecodeRequest?
  priv mut attempted : Bool
  priv mut closed : Bool
}

///|
pub fn Request::new() -> Self {
  {
    pipeline: @joint_diffusion_device.PipelineProgram::new(),
    video: None,
    audio: None,
    attempted: false,
    closed: false,
  }
}

///|
/// Preparation-only composition. Conditioning must be topologically ordered
/// (vision -> text -> refiner / reference encoders -> row assembly). Predictions
/// should come from PredictionMetadata::bind_predictions, not a replayed step.
/// Initial validation failure transfers nothing. After it succeeds, all supplied
/// stages belong to this request even when device queue creation fails.
pub fn Request::prepare(
  self : Self,
  context : @device.Context,
  stream : @device.Stream,
  plan : @joint_diffusion_loop.PreparedPlan,
  storage : @joint_diffusion_device.LatentStorage,
  conditioning : ArrayView[@joint_diffusion_execution.PreparedStage],
  predictions : ArrayView[ReadOnlyArray[@device.OrderedKernelLaunch]],
  video : @video.VideoDecodeRequest,
  audio : @blocks.AudioDecodeRequest,
  video_update~ : @device.Function,
  audio_update~ : @device.Function,
  video_block_size~ : Int,
  audio_block_size~ : Int,
) -> Unit raise {
  guard !self.attempted &&
    !self.closed &&
    !conditioning.is_empty() &&
    predictions.length() == plan.steps().length() &&
    storage.matches_prepared(plan) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  for launches in predictions {
    guard !launches.is_empty() else {
      raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
    }
  }
  guard storage.with_region(VideoState, (allocation, offset, bytes) => {
      video.matches_input(plan, allocation, offset, bytes)
    }) &&
    storage.with_region(AudioState, (allocation, offset, bytes) => {
      audio.matches_input(plan, allocation, offset, bytes)
    }) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.attempted = true
  self.pipeline.prepare_stages(
    context,
    stream,
    plan,
    storage,
    conditioning,
    predictions,
    @joint_diffusion_execution.PreparedStage::own(video),
    @joint_diffusion_execution.PreparedStage::own(audio),
    video_update~,
    audio_update~,
    video_block_size~,
    audio_block_size~,
  )
  self.video = Some(video)
  self.audio = Some(audio)
}

///|
/// One prepared conditioning, paired denoise, or decoder stage per call. No
/// request-time compilation, weight upload or per-step launch construction.
pub fn Request::advance(
  self : Self,
) -> @joint_diffusion_execution.PipelineStage raise {
  guard !self.closed else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue)
  }
  self.pipeline.advance()
}

///|
pub fn Request::cancel(self : Self) -> Unit {
  self.pipeline.cancel()
}

///|
pub fn Request::outputs_ready(self : Self) -> Bool {
  !self.closed && self.pipeline.outputs_ready()
}

///|
/// Both modalities must finish before either is published as a joint result.
pub fn[T] Request::with_audio(
  self : Self,
  consume : (@device.Allocation) -> T raise @device.DeviceError,
) -> T raise @device.DeviceError {
  guard self.outputs_ready() && self.audio is Some(audio) else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  audio.with_output(consume)
}

///|
/// Actual samples per channel, including the reference decoder's rounded
/// latent duration. Do not substitute the requested video-duration sample count.
pub fn Request::waveform_samples(
  self : Self,
) -> Int64 raise @device.DeviceError {
  guard self.outputs_ready() && self.audio is Some(audio) else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  audio.waveform_samples().to_int64()
}

///|
pub fn[T] Request::with_video(
  self : Self,
  consume : (@device.Allocation) -> T raise @device.DeviceError,
) -> T raise @device.DeviceError {
  guard self.outputs_ready() && self.video is Some(video) else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  video.with_output(consume)
}

///|
/// Release downstream stages before upstream conditioning. Failed cleanup is
/// retryable and never unlocks output or permits a second submission.
pub fn Request::close(self : Self) -> Unit raise {
  self.closed = true
  self.pipeline.close()
  self.video = None
  self.audio = None
}

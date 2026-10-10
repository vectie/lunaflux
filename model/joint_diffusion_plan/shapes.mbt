///|
pub fn JointDiffusionShapeContract::new(
  frames_per_second~ : Int,
  min_duration_seconds~ : Int,
  max_duration_seconds~ : Int,
  frames_per_chunk~ : Int,
  latents_per_chunk~ : Int,
  canvas_multiple~ : Int,
  max_canvas_pixels~ : Int64,
  min_aspect_numerator~ : Int,
  min_aspect_denominator~ : Int,
  max_aspect_numerator~ : Int,
  max_aspect_denominator~ : Int,
  video_latent_channels~ : Int,
  video_spatial_compression~ : Int,
  audio_latent_channels~ : Int,
  audio_latents_per_second~ : Int,
  audio_sample_rate~ : Int,
  audio_output_channels~ : Int,
  audio_vae_latent_dimension~ : Int,
  audio_vae_decoder_dimension~ : Int,
  min_inference_steps~ : Int,
  max_inference_steps~ : Int,
  video_patch_temporal? : Int = 1,
  video_patch_height? : Int = 1,
  video_patch_width? : Int = 1,
  audio_latent_streams? : Int = 1,
  audio_output_length? : JointAudioOutputLength = FrameDuration,
) -> JointDiffusionShapeContract raise JointDiffusionPlanError {
  if frames_per_second <= 0 ||
    frames_per_second > 1000000 ||
    min_duration_seconds <= 0 ||
    max_duration_seconds < min_duration_seconds ||
    max_duration_seconds > 1000000 ||
    frames_per_chunk <= 0 ||
    latents_per_chunk <= 0 ||
    latents_per_chunk >= frames_per_chunk ||
    canvas_multiple <= 0 ||
    max_canvas_pixels <= 0L ||
    max_canvas_pixels > 1000000000000L ||
    min_aspect_numerator <= 0 ||
    min_aspect_denominator <= 0 ||
    max_aspect_numerator <= 0 ||
    max_aspect_denominator <= 0 ||
    min_aspect_numerator > 1000000 ||
    min_aspect_denominator > 1000000 ||
    max_aspect_numerator > 1000000 ||
    max_aspect_denominator > 1000000 ||
    min_aspect_numerator * max_aspect_denominator >
    max_aspect_numerator * min_aspect_denominator ||
    video_latent_channels <= 0 ||
    video_spatial_compression <= 0 ||
    canvas_multiple % video_spatial_compression != 0 ||
    audio_latent_channels <= 0 ||
    audio_latent_streams <= 0 ||
    audio_latents_per_second <= 0 ||
    audio_latents_per_second > 1000000 ||
    audio_sample_rate <= 0 ||
    audio_sample_rate > 1000000 ||
    audio_output_channels <= 0 ||
    audio_vae_latent_dimension <= 0 ||
    audio_vae_decoder_dimension <= 0 ||
    min_inference_steps < 2 ||
    max_inference_steps < min_inference_steps ||
    max_inference_steps > 1000000 ||
    video_patch_temporal <= 0 ||
    video_patch_height <= 0 ||
    video_patch_width <= 0 ||
    video_patch_temporal > 1024 ||
    video_patch_height > 1024 ||
    video_patch_width > 1024 ||
    video_patch_height > 1000000 / video_spatial_compression ||
    video_patch_width > 1000000 / video_spatial_compression ||
    canvas_multiple % (video_spatial_compression * video_patch_height) != 0 ||
    canvas_multiple % (video_spatial_compression * video_patch_width) != 0 {
    raise InvalidShapeContract
  }
  if audio_output_length is LatentStride(stride) &&
    (
      stride <= 0 ||
      stride.to_int64() * audio_latents_per_second.to_int64() !=
      audio_sample_rate.to_int64()
    ) {
    raise InvalidShapeContract
  }
  {
    frames_per_second,
    min_duration_seconds,
    max_duration_seconds,
    frames_per_chunk,
    latents_per_chunk,
    canvas_multiple,
    max_canvas_pixels,
    min_aspect_numerator,
    min_aspect_denominator,
    max_aspect_numerator,
    max_aspect_denominator,
    video_latent_channels,
    video_spatial_compression,
    audio_latent_channels,
    audio_latent_streams,
    audio_latents_per_second,
    audio_sample_rate,
    audio_output_channels,
    audio_output_length,
    audio_vae_latent_dimension,
    audio_vae_decoder_dimension,
    min_inference_steps,
    max_inference_steps,
    video_patch_temporal,
    video_patch_height,
    video_patch_width,
  }
}

///|
pub fn JointDiffusionShapeContract::frames_per_second(self : Self) -> Int {
  self.frames_per_second
}

///|
pub fn JointDiffusionShapeContract::min_duration_seconds(self : Self) -> Int {
  self.min_duration_seconds
}

///|
pub fn JointDiffusionShapeContract::max_duration_seconds(self : Self) -> Int {
  self.max_duration_seconds
}

///|
pub fn JointDiffusionShapeContract::frames_per_chunk(self : Self) -> Int {
  self.frames_per_chunk
}

///|
pub fn JointDiffusionShapeContract::latents_per_chunk(self : Self) -> Int {
  self.latents_per_chunk
}

///|
pub fn JointDiffusionShapeContract::canvas_multiple(self : Self) -> Int {
  self.canvas_multiple
}

///|
pub fn JointDiffusionShapeContract::max_canvas_pixels(self : Self) -> Int64 {
  self.max_canvas_pixels
}

///|
pub fn JointDiffusionShapeContract::min_aspect_numerator(self : Self) -> Int {
  self.min_aspect_numerator
}

///|
pub fn JointDiffusionShapeContract::min_aspect_denominator(self : Self) -> Int {
  self.min_aspect_denominator
}

///|
pub fn JointDiffusionShapeContract::max_aspect_numerator(self : Self) -> Int {
  self.max_aspect_numerator
}

///|
pub fn JointDiffusionShapeContract::max_aspect_denominator(self : Self) -> Int {
  self.max_aspect_denominator
}

///|
pub fn JointDiffusionShapeContract::video_latent_channels(self : Self) -> Int {
  self.video_latent_channels
}

///|
pub fn JointDiffusionShapeContract::video_spatial_compression(
  self : Self,
) -> Int {
  self.video_spatial_compression
}

///|
pub fn JointDiffusionShapeContract::audio_latent_channels(self : Self) -> Int {
  self.audio_latent_channels
}

///|
pub fn JointDiffusionShapeContract::audio_latents_per_second(
  self : Self,
) -> Int {
  self.audio_latents_per_second
}

///|
pub fn JointDiffusionShapeContract::audio_sample_rate(self : Self) -> Int {
  self.audio_sample_rate
}

///|
pub fn JointDiffusionShapeContract::audio_output_channels(self : Self) -> Int {
  self.audio_output_channels
}

///|
pub fn JointDiffusionShapeContract::audio_output_length(
  self : Self,
) -> JointAudioOutputLength {
  self.audio_output_length
}

///|
pub fn JointDiffusionShapeContract::audio_vae_latent_dimension(
  self : Self,
) -> Int {
  self.audio_vae_latent_dimension
}

///|
pub fn JointDiffusionShapeContract::audio_vae_decoder_dimension(
  self : Self,
) -> Int {
  self.audio_vae_decoder_dimension
}

///|
pub fn JointDiffusionShapeContract::min_inference_steps(self : Self) -> Int {
  self.min_inference_steps
}

///|
pub fn JointDiffusionShapeContract::max_inference_steps(self : Self) -> Int {
  self.max_inference_steps
}

///|
pub fn JointDiffusionShapeContract::video_patch_temporal(self : Self) -> Int {
  self.video_patch_temporal
}

///|
pub fn JointDiffusionShapeContract::video_patch_height(self : Self) -> Int {
  self.video_patch_height
}

///|
pub fn JointDiffusionShapeContract::video_patch_width(self : Self) -> Int {
  self.video_patch_width
}

///|
fn JointDiffusionShapeContract::align_frames(
  self : Self,
  requested : Int,
) -> Int raise JointDiffusionPlanError {
  if requested <= 0 {
    raise InvalidFrameCount
  }
  let max_frames = self.frames_per_second * self.max_duration_seconds
  if requested > max_frames {
    raise InvalidFrameCount
  }
  let remainder = requested % self.frames_per_chunk
  let delta = if remainder <= self.latents_per_chunk {
    self.latents_per_chunk - remainder
  } else {
    self.frames_per_chunk - remainder + self.latents_per_chunk
  }
  requested + delta
}

///|
pub fn JointDiffusionShapeContract::resolve_request(
  self : Self,
  height~ : Int,
  width~ : Int,
  requested_frames~ : Int,
  inference_steps~ : Int,
) -> JointDiffusionRequestShape raise JointDiffusionPlanError {
  if height <= 0 ||
    width <= 0 ||
    height % self.canvas_multiple != 0 ||
    width % self.canvas_multiple != 0 {
    raise InvalidCanvas
  }
  let height64 = height.to_int64()
  let width64 = width.to_int64()
  if height64 > self.max_canvas_pixels ||
    width64 > self.max_canvas_pixels ||
    height64 > self.max_canvas_pixels / width64 ||
    width64 * self.min_aspect_denominator.to_int64() <
    height64 * self.min_aspect_numerator.to_int64() ||
    width64 * self.max_aspect_denominator.to_int64() >
    height64 * self.max_aspect_numerator.to_int64() {
    raise InvalidCanvas
  }
  if inference_steps < self.min_inference_steps ||
    inference_steps > self.max_inference_steps {
    raise InvalidInferenceStepCount
  }
  let frames = self.align_frames(requested_frames)
  if frames < self.frames_per_second * self.min_duration_seconds ||
    frames > self.frames_per_second * self.max_duration_seconds {
    raise InvalidFrameCount
  }
  let chunk_count = (frames - self.latents_per_chunk) / self.frames_per_chunk
  let video_latent_frames = chunk_count * self.latents_per_chunk + 2
  let audio_latent_frames = (
      frames * self.audio_latents_per_second + self.frames_per_second / 2
    ) /
    self.frames_per_second
  let audio_samples = (
      frames.to_int64() * self.audio_sample_rate.to_int64() +
      (self.frames_per_second / 2).to_int64()
    ) /
    self.frames_per_second.to_int64()
  let audio_output_samples = match self.audio_output_length {
    FrameDuration => audio_samples
    LatentStride(stride) => audio_latent_frames.to_int64() * stride.to_int64()
  }
  guard audio_latent_frames > 0 &&
    self.audio_latent_streams <= 2147483647 / audio_latent_frames else {
    raise InvalidFrameCount
  }
  let audio_token_rows = audio_latent_frames * self.audio_latent_streams
  {
    height,
    width,
    frames,
    inference_steps,
    model_evaluations: inference_steps,
    video_latent_frames,
    video_latent_height: height / self.video_spatial_compression,
    video_latent_width: width / self.video_spatial_compression,
    audio_latent_frames,
    audio_token_rows,
    audio_samples,
    audio_output_samples,
  }
}

///|
pub fn JointDiffusionRequestShape::height(self : Self) -> Int {
  self.height
}

///|
pub fn JointDiffusionRequestShape::width(self : Self) -> Int {
  self.width
}

///|
pub fn JointDiffusionRequestShape::frames(self : Self) -> Int {
  self.frames
}

///|
pub fn JointDiffusionRequestShape::inference_steps(self : Self) -> Int {
  self.inference_steps
}

///|
pub fn JointDiffusionRequestShape::model_evaluations(self : Self) -> Int {
  self.model_evaluations
}

///|
pub fn JointDiffusionRequestShape::video_latent_frames(self : Self) -> Int {
  self.video_latent_frames
}

///|
pub fn JointDiffusionRequestShape::video_latent_height(self : Self) -> Int {
  self.video_latent_height
}

///|
pub fn JointDiffusionRequestShape::video_latent_width(self : Self) -> Int {
  self.video_latent_width
}

///|
pub fn JointDiffusionRequestShape::audio_latent_frames(self : Self) -> Int {
  self.audio_latent_frames
}

///|
/// Per-stream time is distinct from the concatenated stream-major DiT rows.
pub fn JointDiffusionRequestShape::audio_token_rows(self : Self) -> Int {
  self.audio_token_rows
}

///|
pub fn JointDiffusionShapeContract::audio_latent_streams(self : Self) -> Int {
  self.audio_latent_streams
}

///|
pub fn JointDiffusionRequestShape::audio_samples(self : Self) -> Int64 {
  self.audio_samples
}

///|
/// Exact samples per output channel, including declared decoder rounding.
/// `audio_samples()` instead describes the nominal video-frame duration.
pub fn JointDiffusionRequestShape::audio_output_samples(self : Self) -> Int64 {
  self.audio_output_samples
}

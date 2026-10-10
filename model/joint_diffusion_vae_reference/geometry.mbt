///|
fn checked_product(
  values : ArrayView[Int64],
) -> Int64 raise JointVaeDecodeReferenceError {
  let mut product = 1L
  for value in values {
    if value <= 0L || product > 9223372036854775807L / value {
      raise GeometryOverflow
    }
    product *= value
  }
  product
}

///|
fn checked_conv_transpose_output_time(
  input_time : Int,
  kernel_size : Int,
  stride : Int,
  padding : Int,
) -> Int64 raise JointVaeDecodeReferenceError {
  if input_time <= 0 || kernel_size <= 0 || stride <= 0 || padding < 0 {
    raise GeometryOverflow
  }
  let maximum = 9223372036854775807L
  let input_span = input_time.to_int64() - 1L
  let stride64 = stride.to_int64()
  if input_span > maximum / stride64 {
    raise GeometryOverflow
  }
  let expanded = input_span * stride64
  let kernel64 = kernel_size.to_int64()
  if kernel64 > maximum - expanded {
    raise GeometryOverflow
  }
  let padded_extent = expanded + kernel64
  let padding64 = padding.to_int64()
  if padding64 > maximum / 2L {
    raise GeometryOverflow
  }
  let trim = padding64 * 2L
  if trim >= padded_extent {
    raise GeometryOverflow
  }
  padded_extent - trim
}

///|
fn request_matches_plan(
  plan : @joint_diffusion_plan.JointDiffusionPlan,
  request : @joint_diffusion_plan.JointDiffusionRequestShape,
) -> Bool {
  let reconstructed = plan
    .shapes()
    .resolve_request(
      height=request.height(),
      width=request.width(),
      requested_frames=request.frames(),
      inference_steps=request.inference_steps(),
    ) catch {
      _ => return false
    }
  reconstructed == request
}

///|
pub fn JointVaeDecodeReferenceLimits::new(
  maximum_video_latent_elements~ : Int64,
  maximum_video_output_sites~ : Int64,
  maximum_audio_latent_elements~ : Int64,
  maximum_audio_output_elements~ : Int64,
) -> Self raise JointVaeDecodeReferenceError {
  if maximum_video_latent_elements <= 0L ||
    maximum_video_output_sites <= 0L ||
    maximum_audio_latent_elements <= 0L ||
    maximum_audio_output_elements <= 0L {
    raise InvalidLimit
  }
  {
    maximum_video_latent_elements,
    maximum_video_output_sites,
    maximum_audio_latent_elements,
    maximum_audio_output_elements,
  }
}

///|
fn video_gaps() -> ReadOnlyArray[JointVaeDecodeGap] {
  ReadOnlyArray::from_array([
    VideoOutputChannels,
    VideoDecoderDTypeCast,
    VideoPostQuantConvolution,
    VideoSpatialTiling,
    VideoTemporalChunking,
    VideoTransformerDecode,
    VideoOverlapBlending,
    VideoPixelPostprocessing,
    VideoNormalization,
    VideoActivation,
    VideoConvolutionKernel,
    VideoPadding,
    VideoChannelSchedule,
    VideoTemporalUpsample,
    VideoSpatialUpsample,
    VideoResidualPath,
    VideoFinalProjection,
  ])
}

///|
fn audio_gaps() -> ReadOnlyArray[JointVaeDecodeGap] {
  ReadOnlyArray::from_array([
    AudioTemporalUpsample,
    AudioResidualPath,
    AudioNormalization,
    AudioActivation,
    AudioConvolutionKernel,
    AudioPadding,
    AudioChannelSchedule,
    AudioFinalProjection,
  ])
}

///|
fn gap_tag(gap : JointVaeDecodeGap) -> Int {
  match gap {
    VideoOutputChannels => 1
    VideoDecoderDTypeCast => 2
    VideoPostQuantConvolution => 3
    VideoSpatialTiling => 4
    VideoTemporalChunking => 5
    VideoTransformerDecode => 6
    VideoOverlapBlending => 7
    VideoPixelPostprocessing => 8
    VideoNormalization => 9
    VideoActivation => 10
    VideoConvolutionKernel => 11
    VideoPadding => 12
    VideoChannelSchedule => 13
    VideoTemporalUpsample => 14
    VideoSpatialUpsample => 15
    VideoResidualPath => 16
    VideoFinalProjection => 17
    AudioNormalization => 18
    AudioActivation => 19
    AudioConvolutionKernel => 20
    AudioPadding => 21
    AudioChannelSchedule => 22
    AudioTemporalUpsample => 23
    AudioResidualPath => 24
    AudioFinalProjection => 25
  }
}

///|
fn gap_digest(
  plan_digest : @spec.PlanDigest,
  video : ReadOnlyArray[JointVaeDecodeGap],
  audio : ReadOnlyArray[JointVaeDecodeGap],
) -> JointVaeDecodeGapDigest {
  let canonical = StringBuilder()
  canonical <+ "lunaflux.joint-vae-unavailable.v1|plan=\{plan_digest.as_hex()}"
  for gap in video {
    canonical <+ "|video=\{gap_tag(gap)}"
  }
  for gap in audio {
    canonical <+ "|audio=\{gap_tag(gap)}"
  }
  {
    hex: @crypto.bytes_to_hex_string(
      @crypto.sha256(@utf8.encode(canonical.to_string())),
    ),
  }
}

///|
/// Resolves only geometry explicitly carried by the generic plan and request.
pub fn assess_joint_vae_decode(
  plan : @joint_diffusion_plan.JointDiffusionPlan,
  request : @joint_diffusion_plan.JointDiffusionRequestShape,
  limits~ : JointVaeDecodeReferenceLimits,
) -> JointVaeDecodeAssessment raise JointVaeDecodeReferenceError {
  if plan.schema_version() != 1 ||
    !plan.components().contains(VideoVae) ||
    !plan.components().contains(AudioVae) {
    raise InvalidPlan
  }
  if !request_matches_plan(plan, request) {
    raise InvalidRequest
  }
  let shapes = plan.shapes()
  let video_latent_elements = checked_product([
    shapes.video_latent_channels().to_int64(),
    request.video_latent_frames().to_int64(),
    request.video_latent_height().to_int64(),
    request.video_latent_width().to_int64(),
  ])
  let video_output_sites = checked_product([
    request.frames().to_int64(),
    request.height().to_int64(),
    request.width().to_int64(),
  ])
  let audio_latent_elements = checked_product([
    shapes.audio_latent_channels().to_int64(),
    request.audio_token_rows().to_int64(),
  ])
  let audio_output_elements = checked_product([
    request.audio_output_samples(),
    shapes.audio_output_channels().to_int64(),
  ])
  if video_latent_elements > limits.maximum_video_latent_elements ||
    video_output_sites > limits.maximum_video_output_sites {
    raise GeometryBudgetExceeded(VideoVae)
  }
  if audio_latent_elements > limits.maximum_audio_latent_elements ||
    audio_output_elements > limits.maximum_audio_output_elements {
    raise GeometryBudgetExceeded(AudioVae)
  }
  let geometry : JointVaeDecodeGeometry = {
    plan_digest: plan.plan_digest(),
    video_latent_channels: shapes.video_latent_channels(),
    video_latent_frames: request.video_latent_frames(),
    video_latent_height: request.video_latent_height(),
    video_latent_width: request.video_latent_width(),
    video_latent_elements,
    video_output_frames: request.frames(),
    video_output_height: request.height(),
    video_output_width: request.width(),
    video_output_sites,
    video_spatial_compression: shapes.video_spatial_compression(),
    audio_latent_channels: shapes.audio_latent_channels(),
    audio_latent_frames: request.audio_latent_frames(),
    audio_latent_elements,
    audio_output_samples: request.audio_output_samples(),
    audio_output_channels: shapes.audio_output_channels(),
    audio_output_elements,
  }
  let unavailable_video = video_gaps()
  let unavailable_audio = audio_gaps()
  {
    geometry,
    gap_digest: gap_digest(
      geometry.plan_digest,
      unavailable_video,
      unavailable_audio,
    ),
    video_gaps: unavailable_video,
    audio_gaps: unavailable_audio,
  }
}

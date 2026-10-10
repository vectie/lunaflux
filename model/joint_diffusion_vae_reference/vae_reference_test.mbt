///|
fn vae_content(seed : Char) -> @spec.ContentDigest {
  try! @spec.ContentDigest::from_sha256(String::make(64, seed))
}

///|
fn tiny_vae_plan(
  seed? : Char = '1',
) -> @joint_diffusion_plan.JointDiffusionPlan {
  try! @joint_diffusion_plan.JointDiffusionPlan::new(
    schema_version=1,
    content_digest=vae_content(seed),
    workflows=[TextToAudioVideo],
    stages=[
      @joint_diffusion_plan.JointDiffusionStage::new(
        kind=ResolveRequest,
        presence=Required,
      ),
      @joint_diffusion_plan.JointDiffusionStage::new(
        kind=EncodeTextConditioning,
        presence=Required,
      ),
      @joint_diffusion_plan.JointDiffusionStage::new(
        kind=InitializeJointLatents,
        presence=Required,
      ),
      @joint_diffusion_plan.JointDiffusionStage::new(
        kind=JointAudioVideoDenoise,
        presence=Required,
      ),
      @joint_diffusion_plan.JointDiffusionStage::new(
        kind=DecodeVideo,
        presence=Required,
      ),
      @joint_diffusion_plan.JointDiffusionStage::new(
        kind=DecodeAudio,
        presence=Required,
      ),
    ],
    video_scheduler=@joint_diffusion_plan.JointDiffusionScheduler::new(
      latent=VideoLatent,
      kind=RectifiedFlowEuler,
      shift_numerator=2,
      shift_denominator=1,
    ),
    audio_scheduler=@joint_diffusion_plan.JointDiffusionScheduler::new(
      latent=AudioLatent,
      kind=RectifiedFlowEuler,
      shift_numerator=1,
      shift_denominator=1,
    ),
    shapes=@joint_diffusion_plan.JointDiffusionShapeContract::new(
      frames_per_second=4,
      min_duration_seconds=1,
      max_duration_seconds=2,
      frames_per_chunk=3,
      latents_per_chunk=1,
      canvas_multiple=4,
      max_canvas_pixels=16L,
      min_aspect_numerator=1,
      min_aspect_denominator=1,
      max_aspect_numerator=1,
      max_aspect_denominator=1,
      video_latent_channels=2,
      video_spatial_compression=2,
      audio_latent_channels=2,
      audio_latents_per_second=2,
      audio_sample_rate=8,
      audio_output_channels=1,
      audio_vae_latent_dimension=4,
      audio_vae_decoder_dimension=2,
      min_inference_steps=2,
      max_inference_steps=8,
      video_patch_temporal=1,
      video_patch_height=2,
      video_patch_width=2,
    ),
    conditioning=@joint_diffusion_plan.JointDiffusionConditioningLimits::new(
      max_keyframes=0,
      max_image_references=0,
      max_video_references=0,
      max_audio_references=0,
      max_total_references=0,
      allow_audio_only_references=false,
    ),
    projection=@joint_diffusion_plan.JointDiffusionProjectionContract::mixed_precision_float32_heads(),
    denoiser=@joint_diffusion_plan.JointDiffusionDenoiserShape::new(
      hidden_size=4,
      layer_count=1,
      refiner_layer_count=0,
      attention_head_count=2,
      attention_head_dimension=2,
      feed_forward_dimension=8,
      text_conditioning_dimension=4,
      timestep_frequency_dimension=2,
    ),
  )
}

///|
fn tiny_vae_request(
  plan : @joint_diffusion_plan.JointDiffusionPlan,
) -> @joint_diffusion_plan.JointDiffusionRequestShape {
  try! plan
  .shapes()
  .resolve_request(height=4, width=4, requested_frames=4, inference_steps=5)
}

///|
fn vae_limits(
  maximum_video_latent_elements? : Int64 = 10000000L,
  maximum_video_output_sites? : Int64 = 1000000000L,
  maximum_audio_latent_elements? : Int64 = 10000000L,
  maximum_audio_output_elements? : Int64 = 10000000L,
) -> JointVaeDecodeReferenceLimits {
  try! JointVaeDecodeReferenceLimits::new(
    maximum_video_latent_elements~,
    maximum_video_output_sites~,
    maximum_audio_latent_elements~,
    maximum_audio_output_elements~,
  )
}

///|
test "tiny plan proves exact geometry and every unavailable decode semantic" {
  let plan = tiny_vae_plan()
  let assessment = assess_joint_vae_decode(
    plan,
    tiny_vae_request(plan),
    limits=vae_limits(),
  )
  let geometry = assessment.geometry()
  assert_eq(geometry.plan_digest(), plan.plan_digest())
  assert_eq(
    assessment.gap_digest(),
    assess_joint_vae_decode(plan, tiny_vae_request(plan), limits=vae_limits()).gap_digest(),
  )
  assert_eq(geometry.video_latent_channels(), 2)
  assert_eq(geometry.video_latent_frames(), 3)
  assert_eq(geometry.video_latent_height(), 2)
  assert_eq(geometry.video_latent_width(), 2)
  assert_eq(geometry.video_latent_elements(), 24L)
  assert_eq(geometry.video_output_frames(), 4)
  assert_eq(geometry.video_output_height(), 4)
  assert_eq(geometry.video_output_width(), 4)
  assert_eq(geometry.video_output_sites(), 64L)
  assert_eq(geometry.video_spatial_compression(), 2)
  assert_eq(geometry.audio_latent_channels(), 2)
  assert_eq(geometry.audio_latent_frames(), 2)
  assert_eq(geometry.audio_latent_elements(), 4L)
  assert_eq(geometry.audio_output_samples(), 8L)
  assert_eq(geometry.audio_output_channels(), 1)
  assert_eq(geometry.audio_output_elements(), 8L)
  let expected_video : Array[JointVaeDecodeGap] = [
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
  ]
  let expected_audio : Array[JointVaeDecodeGap] = [
    AudioTemporalUpsample,
    AudioResidualPath,
    AudioNormalization,
    AudioActivation,
    AudioConvolutionKernel,
    AudioPadding,
    AudioChannelSchedule,
    AudioFinalProjection,
  ]
  assert_eq(assessment.video_gap_count(), expected_video.length())
  assert_eq(assessment.audio_gap_count(), expected_audio.length())
  for index, gap in expected_video {
    assert_eq(assessment.video_gap(index), Some(gap))
  }
  for index, gap in expected_audio {
    assert_eq(assessment.audio_gap(index), Some(gap))
  }
  assert_eq(assessment.video_gap(expected_video.length()), None)
  assert_eq(assessment.audio_gap(expected_audio.length()), None)
}

///|
test "finite latent admission succeeds but numeric video and audio decode stay typed unavailable" {
  let plan = tiny_vae_plan()
  let assessment = assess_joint_vae_decode(
    plan,
    tiny_vae_request(plan),
    limits=vae_limits(),
  )
  let video = bind_video_latents(assessment, Array::make(24, 0.25))
  let audio = bind_audio_latents(assessment, [1.0, 2.0, 3.0, 4.0])
  assert_eq(video.component(), VideoVae)
  assert_eq(video.plan_digest(), plan.plan_digest())
  assert_eq(audio.values(), [1.0, 2.0, 3.0, 4.0])
  try decode_one_video_stage(assessment, video) catch {
    DecodeSemanticUnavailable(VideoVae, VideoOutputChannels) => ()
    _ => fail("unexpected video decode gap")
  } noraise {
    _ => fail("video decoder execution was fabricated")
  }
  try decode_one_audio_stage(assessment, audio) catch {
    DecodeSemanticUnavailable(AudioVae, AudioTemporalUpsample) => ()
    _ => fail("unexpected audio decode gap")
  } noraise {
    _ => fail("audio decoder execution was fabricated")
  }
}

///|
test "official MiniMax geometry is plan-bound without importing family code in production" {
  let metadata = @minimax_h3_spec.MiniMaxH3ModelMetadata::from_verified_content(
    @minimax_h3_spec.MiniMaxH3TransformerSpec::official(
      variant=Fl2va,
      checkpoint_format=DiffusersV1,
    ),
    vae_content('9'),
  )
  let plan = try! @minimax_h3.build(metadata)
  let request = plan
    .shapes()
    .resolve_request(
      height=768,
      width=1344,
      requested_frames=120,
      inference_steps=5,
    )
  let geometry = assess_joint_vae_decode(
    plan,
    request,
    limits=vae_limits(maximum_audio_output_elements=331200L),
  ).geometry()
  assert_eq(geometry.plan_digest(), plan.plan_digest())
  assert_eq(geometry.video_latent_channels(), 24)
  assert_eq(geometry.video_latent_frames(), 37)
  assert_eq(geometry.video_latent_height(), 48)
  assert_eq(geometry.video_latent_width(), 84)
  assert_eq(geometry.video_latent_elements(), 3580416L)
  assert_eq(geometry.video_output_frames(), 124)
  assert_eq(geometry.video_output_sites(), 127991808L)
  assert_eq(geometry.audio_latent_channels(), 32)
  assert_eq(geometry.audio_latent_frames(), 207)
  assert_eq(geometry.audio_latent_elements(), 13248L)
  assert_eq(geometry.audio_output_samples(), 165600L)
  assert_eq(geometry.audio_output_channels(), 2)
  assert_eq(geometry.audio_output_elements(), 331200L)
  try
    assess_joint_vae_decode(
      plan,
      request,
      limits=vae_limits(maximum_audio_output_elements=331199L),
    )
  catch {
    GeometryBudgetExceeded(AudioVae) => ()
    _ => fail("unexpected rounded audio output budget error")
  } noraise {
    _ => fail("rounded decoder output exceeded its exact element budget")
  }
}

///|
test "foreign identity nonfinite latent shape and geometry budgets fail closed" {
  let plan = tiny_vae_plan()
  let assessment = assess_joint_vae_decode(
    plan,
    tiny_vae_request(plan),
    limits=vae_limits(),
  )
  try bind_video_latents(assessment, Array::make(23, 0.0)) catch {
    InvalidLatentElementCount(VideoVae) => ()
    _ => fail("unexpected video length error")
  } noraise {
    _ => fail("short video latent admitted")
  }
  let hostile = Array::make(24, 0.0)
  hostile[7] = 1.0 / 0.0
  try bind_video_latents(assessment, hostile) catch {
    NonFiniteLatent(VideoVae, 7) => ()
    _ => fail("unexpected nonfinite latent error")
  } noraise {
    _ => fail("nonfinite video latent admitted")
  }
  let foreign_plan = tiny_vae_plan(seed='2')
  let foreign = assess_joint_vae_decode(
    foreign_plan,
    tiny_vae_request(foreign_plan),
    limits=vae_limits(),
  )
  assert_true(foreign.gap_digest() != assessment.gap_digest())
  let audio = bind_audio_latents(assessment, [0.0, 0.0, 0.0, 0.0])
  try decode_one_audio_stage(foreign, audio) catch {
    LatentIdentityMismatch(AudioVae) => ()
    _ => fail("unexpected foreign identity error")
  } noraise {
    _ => fail("foreign plan latent admitted")
  }
  try
    assess_joint_vae_decode(
      plan,
      tiny_vae_request(plan),
      limits=vae_limits(maximum_video_latent_elements=23L),
    )
  catch {
    GeometryBudgetExceeded(VideoVae) => ()
    _ => fail("unexpected video budget error")
  } noraise {
    _ => fail("video geometry budget exceeded")
  }
  try
    assess_joint_vae_decode(
      plan,
      tiny_vae_request(plan),
      limits=vae_limits(maximum_audio_output_elements=7L),
    )
  catch {
    GeometryBudgetExceeded(AudioVae) => ()
    _ => fail("unexpected audio budget error")
  } noraise {
    _ => fail("audio geometry budget exceeded")
  }
}

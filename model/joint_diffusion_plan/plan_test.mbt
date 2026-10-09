///|
fn test_digest() -> @spec.ContentDigest {
  try! @spec.ContentDigest::from_sha256(
    "abababababababababababababababababababababababababababababababab",
  )
}

///|
fn test_shapes(
  streams? : Int = 1,
) -> @joint_diffusion_plan.JointDiffusionShapeContract {
  try! @joint_diffusion_plan.JointDiffusionShapeContract::new(
    frames_per_second=24,
    min_duration_seconds=5,
    max_duration_seconds=15,
    frames_per_chunk=17,
    latents_per_chunk=5,
    canvas_multiple=32,
    max_canvas_pixels=1032192L,
    min_aspect_numerator=1,
    min_aspect_denominator=4,
    max_aspect_numerator=4,
    max_aspect_denominator=1,
    video_latent_channels=24,
    video_spatial_compression=16,
    audio_latent_channels=32,
    audio_latent_streams=streams,
    audio_latents_per_second=40,
    audio_sample_rate=32000,
    audio_output_channels=2,
    audio_vae_latent_dimension=2048,
    audio_vae_decoder_dimension=1024,
    min_inference_steps=2,
    max_inference_steps=1000,
  )
}

///|
fn test_scheduler(
  latent : @joint_diffusion_plan.JointDiffusionLatentKind,
  shift : Int,
) -> @joint_diffusion_plan.JointDiffusionScheduler {
  try! @joint_diffusion_plan.JointDiffusionScheduler::new(
    latent~,
    kind=RectifiedFlowEuler,
    shift_numerator=shift,
    shift_denominator=1,
  )
}

///|
fn no_conditioning() -> @joint_diffusion_plan.JointDiffusionConditioningLimits {
  try! @joint_diffusion_plan.JointDiffusionConditioningLimits::new(
    max_keyframes=0,
    max_image_references=0,
    max_video_references=0,
    max_audio_references=0,
    max_total_references=0,
    allow_audio_only_references=false,
  )
}

///|
fn test_denoiser() -> @joint_diffusion_plan.JointDiffusionDenoiserShape {
  try! @joint_diffusion_plan.JointDiffusionDenoiserShape::new(
    hidden_size=1024,
    layer_count=8,
    refiner_layer_count=1,
    attention_head_count=8,
    attention_head_dimension=128,
    feed_forward_dimension=4096,
    text_conditioning_dimension=2048,
    timestep_frequency_dimension=256,
  )
}

///|
test "denoiser authenticates exact rational final RMSNorm epsilon" {
  let denoiser = test_denoiser()
  assert_eq(denoiser.final_norm_epsilon_numerator(), 1)
  assert_eq(denoiser.final_norm_epsilon_denominator(), 100000)
  assert_eq(denoiser.attention_rotary_dimension(), 2)
  assert_eq(denoiser.query_key_norm_epsilon_numerator(), 1)
  assert_eq(denoiser.query_key_norm_epsilon_denominator(), 100000)
  assert_eq(denoiser.timestep_embedding_hidden_dimension(), 4)
  assert_eq(denoiser.final_adaln_conditioning_dimension(), 2)
  assert_eq(denoiser.final_norm_epsilon(), 0.00001)
  try
    @joint_diffusion_plan.JointDiffusionDenoiserShape::new(
      hidden_size=1024,
      layer_count=8,
      refiner_layer_count=1,
      attention_head_count=8,
      attention_head_dimension=128,
      feed_forward_dimension=4096,
      text_conditioning_dimension=2048,
      timestep_frequency_dimension=256,
      final_norm_epsilon_numerator=0,
      final_norm_epsilon_denominator=100000,
    )
  catch {
    InvalidShapeContract => ()
    _ => fail("unexpected final RMSNorm epsilon rejection")
  } noraise {
    _ => fail("zero final RMSNorm epsilon was admitted")
  }
}

///|
test "request resolution rejects hostile canvas frame and step envelopes" {
  let shapes = test_shapes()
  try
    shapes.resolve_request(
      height=769,
      width=1344,
      requested_frames=124,
      inference_steps=50,
    )
  catch {
    InvalidCanvas => ()
    _ => fail("unexpected invalid-canvas result")
  } noraise {
    _ => fail("unaligned canvas was admitted")
  }
  try
    shapes.resolve_request(
      height=768,
      width=1344,
      requested_frames=346,
      inference_steps=50,
    )
  catch {
    InvalidFrameCount => ()
    _ => fail("unexpected invalid-frame result")
  } noraise {
    _ => fail("over-duration aligned frame count was admitted")
  }
  try
    shapes.resolve_request(
      height=768,
      width=1344,
      requested_frames=124,
      inference_steps=1,
    )
  catch {
    InvalidInferenceStepCount => ()
    _ => fail("unexpected invalid-step result")
  } noraise {
    _ => fail("one-point sigma schedule was admitted")
  }
}

///|
test "plan rejects reordered stages and crossed schedulers" {
  let stages = [
    @joint_diffusion_plan.JointDiffusionStage::new(
      kind=EncodeTextConditioning,
      presence=Required,
    ),
    @joint_diffusion_plan.JointDiffusionStage::new(
      kind=ResolveRequest,
      presence=Required,
    ),
  ]
  try
    @joint_diffusion_plan.JointDiffusionPlan::new(
      schema_version=1,
      content_digest=test_digest(),
      workflows=[TextToAudioVideo, FirstLastFrameToAudioVideo],
      stages~,
      video_scheduler=test_scheduler(VideoLatent, 12),
      audio_scheduler=test_scheduler(AudioLatent, 3),
      shapes=test_shapes(),
      conditioning=no_conditioning(),
      denoiser=test_denoiser(),
      projection=JointDiffusionProjectionContract::mixed_precision_float32_heads(),
    )
  catch {
    InvalidStageSequence => ()
    _ => fail("unexpected reordered-stage result")
  } noraise {
    _ => fail("reordered diffusion stages were admitted")
  }
  let valid_stages = [
    @joint_diffusion_plan.JointDiffusionStage::new(
      kind=ResolveRequest,
      presence=Required,
    ),
    @joint_diffusion_plan.JointDiffusionStage::new(
      kind=EncodeTextConditioning,
      presence=Required,
    ),
    @joint_diffusion_plan.JointDiffusionStage::new(
      kind=EncodeReferenceConditioning,
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
  ]
  try
    @joint_diffusion_plan.JointDiffusionPlan::new(
      schema_version=1,
      content_digest=test_digest(),
      workflows=[ReferenceToAudioVideo],
      stages=valid_stages,
      video_scheduler=test_scheduler(AudioLatent, 12),
      audio_scheduler=test_scheduler(VideoLatent, 3),
      shapes=test_shapes(),
      conditioning=no_conditioning(),
      denoiser=test_denoiser(),
      projection=JointDiffusionProjectionContract::mixed_precision_float32_heads(),
    )
  catch {
    InvalidScheduler => ()
    _ => fail("unexpected crossed-scheduler result")
  } noraise {
    _ => fail("crossed modality schedulers were admitted")
  }
}

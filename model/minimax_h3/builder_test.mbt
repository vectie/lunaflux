///|
fn h3_metadata(
  variant : @minimax_h3_spec.MiniMaxH3Variant,
  checkpoint_format? : @minimax_h3_spec.MiniMaxH3CheckpointFormat = DiffusersV1,
) -> @minimax_h3_spec.MiniMaxH3ModelMetadata {
  let spec = @minimax_h3_spec.MiniMaxH3TransformerSpec::official(
    variant~,
    checkpoint_format~,
  )
  @minimax_h3_spec.MiniMaxH3ModelMetadata::from_verified_content(
    spec,
    try! @spec.ContentDigest::from_sha256(
      "4444444444444444444444444444444444444444444444444444444444444444",
    ),
  )
}

///|
test "MiniMax H3 declares the complete official pipeline requirements" {
  let profile = @minimax_h3.requirements()
  assert_eq(profile.family(), MiniMaxH3)
  assert_eq(profile.architecture(), "MiniMaxH3DiTModel")
  assert_eq(profile.model_type(), "minimax_h3")
  assert_eq(profile.schema_version(), 1)
  assert_eq(profile.workload(), AudioVideoDiffusion)
  assert_eq(profile.capabilities(), [
    TextConditioningEncoder,
    VisionEncoder,
    MultimodalProjection,
    DiffusionDenoising,
    VideoVae,
    AudioVae,
    ReferenceConditioning,
  ])
}

///|
test "MiniMax H3 backend admission reports every missing semantic" {
  let partial = try! @family_contract.ModelBackendCapabilities::new([
    DiffusionDenoising,
    VideoVae,
  ])
  match @minimax_h3.admit_backend(partial) {
    MissingCapabilities(profile, missing) => {
      assert_eq(profile.family(), MiniMaxH3)
      assert_eq(missing, [
        TextConditioningEncoder,
        VisionEncoder,
        MultimodalProjection,
        AudioVae,
        ReferenceConditioning,
      ])
    }
    SemanticCapabilitiesAvailable(_) =>
      fail("partial MiniMax H3 backend was admitted")
  }
  let complete = try! @family_contract.ModelBackendCapabilities::new([
    TextConditioningEncoder,
    VisionEncoder,
    MultimodalProjection,
    DiffusionDenoising,
    VideoVae,
    AudioVae,
    ReferenceConditioning,
  ])
  assert_true(
    @minimax_h3.admit_backend(complete) is SemanticCapabilitiesAvailable(_),
  )
}

///|
test "FL2VA builds text and optional keyframe joint diffusion stages" {
  let plan = @minimax_h3.build(h3_metadata(Fl2va))
  assert_eq(plan.workflows(), [TextToAudioVideo, FirstLastFrameToAudioVideo])
  assert_eq(plan.stages().map(stage => stage.kind()), [
    ResolveRequest,
    EncodeTextConditioning,
    EncodeKeyframeConditioning,
    InitializeJointLatents,
    JointAudioVideoDenoise,
    DecodeVideo,
    DecodeAudio,
  ])
  assert_eq(plan.stages()[2].presence(), Optional)
  assert_eq(plan.conditioning().max_keyframes(), 2)
  assert_eq(plan.conditioning().max_total_references(), 0)
}

///|
test "Ref2VA builds ordered multimodal reference stages and limits" {
  let plan = @minimax_h3.build(
    h3_metadata(Ref2va, checkpoint_format=OriginalV1),
  )
  assert_eq(plan.workflows(), [ReferenceToAudioVideo])
  assert_eq(plan.stages().map(stage => stage.kind()), [
    ResolveRequest,
    EncodeTextConditioning,
    EncodeReferenceConditioning,
    InitializeJointLatents,
    JointAudioVideoDenoise,
    DecodeVideo,
    DecodeAudio,
  ])
  let limits = plan.conditioning()
  assert_eq(limits.max_image_references(), 9)
  assert_eq(limits.max_video_references(), 3)
  assert_eq(limits.max_audio_references(), 3)
  assert_eq(limits.max_total_references(), 12)
  assert_false(limits.allow_audio_only_references())
}

///|
test "MiniMax H3 resolves released canvas frame and latent geometry" {
  let plan = @minimax_h3.build(h3_metadata(Fl2va))
  let shape = plan
    .shapes()
    .resolve_request(
      height=768,
      width=1344,
      requested_frames=120,
      inference_steps=50,
    )
  assert_eq(shape.frames(), 124)
  assert_eq(shape.video_latent_frames(), 37)
  assert_eq(shape.video_latent_height(), 48)
  assert_eq(shape.video_latent_width(), 84)
  assert_eq(shape.audio_latent_frames(), 207)
  assert_eq(plan.shapes().audio_latent_streams(), 2)
  assert_eq(shape.audio_token_rows(), 414)
  assert_eq(shape.audio_samples(), 165333L)
  assert_eq(shape.audio_output_samples(), 165600L)
  assert_eq(plan.shapes().audio_output_length(), LatentStride(800))
  assert_eq(shape.model_evaluations(), 50)
  assert_eq(plan.video_scheduler().shift(), 12.0)
  assert_eq(plan.audio_scheduler().shift(), 3.0)
  assert_eq(plan.video_scheduler().training_timestep_count(), 1000)
  assert_eq(plan.audio_scheduler().training_timestep_count(), 1000)
  assert_eq(plan.shapes().video_patch_temporal(), 1)
  assert_eq(plan.shapes().video_patch_height(), 2)
  assert_eq(plan.shapes().video_patch_width(), 2)
  assert_eq(plan.projection().storage(), Float32InputWeightBiasBFloat16Output)
  assert_eq(
    plan.projection().video_layout(),
    BatchFrameHeightWidthChannelTemporalHeightWidth,
  )
  assert_eq(plan.projection().audio_layout(), ChannelMajorTime)
  assert_true(plan.packed_joint_sequence())
  assert_true(plan.guidance_distilled())
  assert_true(plan.conditioning_draws_precede_output_draws())
  let denoiser = plan.denoiser()
  assert_eq(denoiser.hidden_size(), 5376)
  assert_eq(denoiser.layer_count(), 50)
  assert_eq(denoiser.refiner_layer_count(), 2)
  assert_eq(denoiser.attention_head_count(), 56)
  assert_eq(denoiser.attention_width(), 7168)
  assert_eq(denoiser.attention_rotary_dimension(), 96)
  assert_eq(denoiser.query_key_norm_epsilon_numerator(), 1)
  assert_eq(denoiser.query_key_norm_epsilon_denominator(), 100000)
  assert_eq(denoiser.feed_forward_dimension(), 14336)
  assert_eq(denoiser.text_conditioning_dimension(), 5120)
  assert_eq(denoiser.timestep_frequency_dimension(), 256)
  assert_eq(denoiser.timestep_embedding_hidden_dimension(), 5376)
  assert_eq(denoiser.final_adaln_conditioning_dimension(), 2688)
}

///|
test "MiniMax H3 plan identity is deterministic and partition-sensitive" {
  let first = @minimax_h3.build(h3_metadata(Fl2va))
  let repeated = @minimax_h3.build(h3_metadata(Fl2va))
  let reference = @minimax_h3.build(h3_metadata(Ref2va))
  assert_eq(first.plan_digest(), repeated.plan_digest())
  assert_true(first.plan_digest() != reference.plan_digest())
}

///|
test "MiniMax H3 projects exact identity into joint diffusion workload" {
  let metadata = h3_metadata(Ref2va, checkpoint_format=OriginalV1)
  let joint = @minimax_h3.build(metadata)
  let workload = @minimax_h3.build_workload_execution_plan(metadata)
  assert_eq(workload.kind(), JointAudioVideoDiffusion)
  assert_eq(workload.schema_version(), joint.schema_version())
  assert_eq(workload.identity().content(), joint.content_digest())
  assert_eq(workload.identity().plan(), joint.plan_digest())
  assert_eq(workload.decoder_layer_count(), None)
  assert_eq(workload.maximum_token_context(), None)
}

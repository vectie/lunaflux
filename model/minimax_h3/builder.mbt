///|
fn video_scheduler() -> @joint_diffusion_plan.JointDiffusionScheduler {
  try! @joint_diffusion_plan.JointDiffusionScheduler::new(
    latent=VideoLatent,
    kind=RectifiedFlowEuler,
    shift_numerator=12,
    shift_denominator=1,
    training_timestep_count=1000,
  )
}

///|
fn audio_scheduler() -> @joint_diffusion_plan.JointDiffusionScheduler {
  try! @joint_diffusion_plan.JointDiffusionScheduler::new(
    latent=AudioLatent,
    kind=RectifiedFlowEuler,
    shift_numerator=3,
    shift_denominator=1,
    training_timestep_count=1000,
  )
}

///|
fn shape_contract() -> @joint_diffusion_plan.JointDiffusionShapeContract {
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
    audio_latent_streams=2,
    audio_latents_per_second=40,
    audio_sample_rate=32000,
    audio_output_channels=2,
    audio_vae_latent_dimension=2048,
    audio_vae_decoder_dimension=1024,
    min_inference_steps=2,
    // This is a LunaFlux admission ceiling, not a checkpoint recommendation.
    // It makes scheduler storage and launch counts finitely plannable.
    max_inference_steps=1000,
    video_patch_temporal=1,
    video_patch_height=2,
    video_patch_width=2,
  )
}

///|
fn fl2va_stages() -> Array[@joint_diffusion_plan.JointDiffusionStage] {
  [
    @joint_diffusion_plan.JointDiffusionStage::new(
      kind=ResolveRequest,
      presence=Required,
    ),
    @joint_diffusion_plan.JointDiffusionStage::new(
      kind=EncodeTextConditioning,
      presence=Required,
    ),
    @joint_diffusion_plan.JointDiffusionStage::new(
      kind=EncodeKeyframeConditioning,
      presence=Optional,
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
}

///|
fn ref2va_stages() -> Array[@joint_diffusion_plan.JointDiffusionStage] {
  [
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
}

///|
/// Builds the exact startup execution plan for one released MiniMax H3
/// checkpoint partition.
///
/// This is intentionally a joint diffusion plan, not a decoder `ModelPlan`.
/// It proves pipeline topology, scheduler identity, component requirements,
/// and bounded request geometry. Weight materialization and device execution
/// remain separate admission layers.
pub fn build(
  metadata : @minimax_h3_spec.MiniMaxH3ModelMetadata,
) -> @joint_diffusion_plan.JointDiffusionPlan raise @joint_diffusion_plan.JointDiffusionPlanError {
  let spec = metadata.spec()
  let (workflows, stages, conditioning) = match spec.variant() {
    Fl2va =>
      (
        [
          @joint_diffusion_plan.JointDiffusionWorkflow::TextToAudioVideo,
          FirstLastFrameToAudioVideo,
        ],
        fl2va_stages(),
        @joint_diffusion_plan.JointDiffusionConditioningLimits::new(
          max_keyframes=2,
          max_image_references=0,
          max_video_references=0,
          max_audio_references=0,
          max_total_references=0,
          allow_audio_only_references=false,
        ),
      )
    Ref2va =>
      (
        [ReferenceToAudioVideo],
        ref2va_stages(),
        @joint_diffusion_plan.JointDiffusionConditioningLimits::new(
          max_keyframes=0,
          max_image_references=9,
          max_video_references=3,
          max_audio_references=3,
          max_total_references=12,
          allow_audio_only_references=false,
        ),
      )
  }
  @joint_diffusion_plan.JointDiffusionPlan::new(
    schema_version=spec.schema_version(),
    content_digest=metadata.content_digest(),
    workflows~,
    stages~,
    video_scheduler=video_scheduler(),
    audio_scheduler=audio_scheduler(),
    shapes=shape_contract(),
    conditioning~,
    projection=@joint_diffusion_plan.JointDiffusionProjectionContract::mixed_precision_float32_heads(),
    denoiser=@joint_diffusion_plan.JointDiffusionDenoiserShape::new(
      hidden_size=spec.hidden_size(),
      layer_count=spec.layer_count(),
      refiner_layer_count=spec.refiner_layer_count(),
      attention_head_count=spec.attention_head_count(),
      attention_head_dimension=spec.attention_head_dimension(),
      attention_rotary_dimension=spec.rope_frequency_dimension() * 6,
      query_key_norm_epsilon_numerator=1,
      query_key_norm_epsilon_denominator=100000,
      feed_forward_dimension=spec.ffn_dimension(),
      text_conditioning_dimension=spec.text_dimension(),
      timestep_frequency_dimension=spec.frequency_dimension(),
      timestep_embedding_hidden_dimension=spec.time_embedding_hidden_dimension(),
      final_adaln_conditioning_dimension=spec.time_embedding_dimension(),
      final_norm_epsilon_numerator=1,
      final_norm_epsilon_denominator=100000,
    ),
  )
}

///|
/// Projects the exact MiniMax H3 joint diffusion plan into the immutable,
/// family-neutral startup workload envelope. Both authenticated content and
/// the complete canonical diffusion plan digest are retained in the identity.
pub fn build_workload_execution_plan(
  metadata : @minimax_h3_spec.MiniMaxH3ModelMetadata,
) -> @workload_execution_plan.WorkloadExecutionPlan raise @joint_diffusion_plan.JointDiffusionPlanError {
  let plan = build(metadata)
  let identity = try! @spec.ModelIdentity::new(
    plan.content_digest(),
    plan.plan_digest(),
  )
  try! @workload_execution_plan.WorkloadExecutionPlan::joint_audio_video_diffusion(
    identity~,
    schema_version=plan.schema_version(),
  )
}

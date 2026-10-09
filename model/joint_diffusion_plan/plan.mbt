///|
fn workflow_tag(value : JointDiffusionWorkflow) -> Int {
  match value {
    TextToAudioVideo => 1
    FirstLastFrameToAudioVideo => 2
    ReferenceToAudioVideo => 3
  }
}

///|
fn stage_tag(value : JointDiffusionStageKind) -> Int {
  match value {
    ResolveRequest => 1
    EncodeTextConditioning => 2
    EncodeKeyframeConditioning => 3
    EncodeReferenceConditioning => 4
    InitializeJointLatents => 5
    JointAudioVideoDenoise => 6
    DecodeVideo => 7
    DecodeAudio => 8
  }
}

///|
fn presence_tag(value : JointDiffusionStagePresence) -> Int {
  match value {
    Required => 1
    Optional => 2
  }
}

///|
pub fn JointDiffusionProjectionContract::mixed_precision_float32_heads() -> Self {
  {
    storage: Float32InputWeightBiasBFloat16Output,
    output_storage: Float32InputWeightBiasFloat32Output,
    video_layout: BatchFrameHeightWidthChannelTemporalHeightWidth,
    audio_layout: ChannelMajorTime,
  }
}

///|
/// Constructs a neighboring BF16 output-head storage contract. This is a
/// valid generic plan vocabulary but is not the official mixed-precision ABI.
pub fn JointDiffusionProjectionContract::mixed_precision_bfloat16_heads() -> Self {
  {
    storage: Float32InputWeightBiasBFloat16Output,
    output_storage: BFloat16InputWeightBiasBFloat16Output,
    video_layout: BatchFrameHeightWidthChannelTemporalHeightWidth,
    audio_layout: ChannelMajorTime,
  }
}

///|
pub fn JointDiffusionProjectionContract::output_storage(
  self : Self,
) -> JointDiffusionOutputProjectionStorage {
  self.output_storage
}

///|
pub fn JointDiffusionProjectionContract::storage(
  self : Self,
) -> JointDiffusionProjectionStorage {
  self.storage
}

///|
pub fn JointDiffusionProjectionContract::video_layout(
  self : Self,
) -> JointDiffusionVideoPatchLayout {
  self.video_layout
}

///|
pub fn JointDiffusionProjectionContract::audio_layout(
  self : Self,
) -> JointDiffusionAudioRowLayout {
  self.audio_layout
}

///|
fn projection_storage_tag(value : JointDiffusionProjectionStorage) -> Int {
  match value {
    Float32InputWeightBiasBFloat16Output => 1
  }
}

///|
fn output_projection_storage_tag(
  value : JointDiffusionOutputProjectionStorage,
) -> Int {
  match value {
    Float32InputWeightBiasFloat32Output => 1
    BFloat16InputWeightBiasBFloat16Output => 2
  }
}

///|
fn video_patch_layout_tag(value : JointDiffusionVideoPatchLayout) -> Int {
  match value {
    BatchFrameHeightWidthChannelTemporalHeightWidth => 1
  }
}

///|
fn audio_row_layout_tag(value : JointDiffusionAudioRowLayout) -> Int {
  match value {
    ChannelMajorTime => 1
  }
}

///|
fn valid_workflows(workflows : ReadOnlyArray[JointDiffusionWorkflow]) -> Bool {
  if workflows.is_empty() || workflows.length() > 3 {
    false
  } else {
    for index, workflow in workflows {
      if index > 0 &&
        workflow_tag(workflows[index - 1]) >= workflow_tag(workflow) {
        break false
      }
    } nobreak {
      true
    }
  }
}

///|
fn valid_stages(
  workflows : ReadOnlyArray[JointDiffusionWorkflow],
  stages : ReadOnlyArray[JointDiffusionStage],
) -> Bool {
  let expected : Array[JointDiffusionStage] = [
    JointDiffusionStage::new(kind=ResolveRequest, presence=Required),
  ]
  expected.push(
    JointDiffusionStage::new(kind=EncodeTextConditioning, presence=Required),
  )
  if workflows.contains(FirstLastFrameToAudioVideo) {
    expected.push(
      JointDiffusionStage::new(
        kind=EncodeKeyframeConditioning,
        presence=Optional,
      ),
    )
  }
  if workflows.contains(ReferenceToAudioVideo) {
    expected.push(
      JointDiffusionStage::new(
        kind=EncodeReferenceConditioning,
        presence=if workflows.length() == 1 { Required } else { Optional },
      ),
    )
  }
  expected.push(
    JointDiffusionStage::new(kind=InitializeJointLatents, presence=Required),
  )
  expected.push(
    JointDiffusionStage::new(kind=JointAudioVideoDenoise, presence=Required),
  )
  expected.push(JointDiffusionStage::new(kind=DecodeVideo, presence=Required))
  expected.push(JointDiffusionStage::new(kind=DecodeAudio, presence=Required))
  stages == ReadOnlyArray::from_array(expected)
}

///|
fn encode_plan_digest(
  schema_version : Int,
  content_digest : @spec.ContentDigest,
  workflows : ReadOnlyArray[JointDiffusionWorkflow],
  stages : ReadOnlyArray[JointDiffusionStage],
  video_scheduler : JointDiffusionScheduler,
  audio_scheduler : JointDiffusionScheduler,
  shapes : JointDiffusionShapeContract,
  conditioning : JointDiffusionConditioningLimits,
  denoiser : JointDiffusionDenoiserShape,
  projection : JointDiffusionProjectionContract,
) -> @spec.PlanDigest {
  let out = StringBuilder()
  out <+
    "lunaflux.joint-diffusion-plan.v1|sv=\{schema_version}|content=\{content_digest.as_hex()}"
  for workflow in workflows {
    out <+ "|w=\{workflow_tag(workflow)}"
  }
  for stage in stages {
    out <+ "|s=\{stage_tag(stage.kind())}:\{presence_tag(stage.presence())}"
  }
  out <+
    "|vs=\{video_scheduler.shift_numerator()}/\{video_scheduler.shift_denominator()}:\{video_scheduler.training_timestep_count()}"
  out <+
    "|as=\{audio_scheduler.shift_numerator()}/\{audio_scheduler.shift_denominator()}:\{audio_scheduler.training_timestep_count()}"
  out <+
    "|fps=\{shapes.frames_per_second()}|fd=\{shapes.min_duration_seconds()}-\{shapes.max_duration_seconds()}"
  out <+
    "|fc=\{shapes.frames_per_chunk()}:\{shapes.latents_per_chunk()}|cm=\{shapes.canvas_multiple()}"
  out <+
    "|mp=\{shapes.max_canvas_pixels()}|vl=\{shapes.video_latent_channels()}:\{shapes.video_spatial_compression()}"
  out <+
    "|aspect=\{shapes.min_aspect_numerator()}/\{shapes.min_aspect_denominator()}-\{shapes.max_aspect_numerator()}/\{shapes.max_aspect_denominator()}"
  out <+
    "|al=\{shapes.audio_latent_channels()}:\{shapes.audio_latents_per_second()}:\{shapes.audio_sample_rate()}:\{shapes.audio_output_channels()}:\{shapes.audio_vae_latent_dimension()}:\{shapes.audio_vae_decoder_dimension()}"
  out <+
    "|steps=\{shapes.min_inference_steps()}-\{shapes.max_inference_steps()}"
  if shapes.audio_latent_streams() != 1 {
    out <+ "|audio-latent-streams=\{shapes.audio_latent_streams()}"
  }
  out <+
    "|vpatch=\{shapes.video_patch_temporal()}:\{shapes.video_patch_height()}:\{shapes.video_patch_width()}"
  out <+
    "|cond=\{conditioning.max_keyframes()}:\{conditioning.max_image_references()}:\{conditioning.max_video_references()}:\{conditioning.max_audio_references()}:\{conditioning.max_total_references()}:\{conditioning.allow_audio_only_references()}"
  out <+
    "|dit=\{denoiser.hidden_size()}:\{denoiser.layer_count()}:\{denoiser.refiner_layer_count()}:\{denoiser.attention_head_count()}:\{denoiser.attention_head_dimension()}:\{denoiser.attention_rotary_dimension()}:\{denoiser.query_key_norm_epsilon_numerator()}/\{denoiser.query_key_norm_epsilon_denominator()}:\{denoiser.feed_forward_dimension()}:\{denoiser.text_conditioning_dimension()}:\{denoiser.timestep_frequency_dimension()}:\{denoiser.timestep_embedding_hidden_dimension()}:\{denoiser.final_adaln_conditioning_dimension()}:\{denoiser.final_norm_epsilon_numerator()}/\{denoiser.final_norm_epsilon_denominator()}"
  out <+
    "|projection=\{projection_storage_tag(projection.storage())}:\{output_projection_storage_tag(projection.output_storage())}:\{video_patch_layout_tag(projection.video_layout())}:\{audio_row_layout_tag(projection.audio_layout())}"
  out <+ "|packed=1|guidance=0|draw-order=conditioning,video,audio"
  try! @spec.PlanDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(@utf8.encode(out.to_string()))),
  )
}

///|
pub fn JointDiffusionPlan::new(
  schema_version~ : Int,
  content_digest~ : @spec.ContentDigest,
  workflows~ : ArrayView[JointDiffusionWorkflow],
  stages~ : ArrayView[JointDiffusionStage],
  video_scheduler~ : JointDiffusionScheduler,
  audio_scheduler~ : JointDiffusionScheduler,
  shapes~ : JointDiffusionShapeContract,
  conditioning~ : JointDiffusionConditioningLimits,
  denoiser~ : JointDiffusionDenoiserShape,
  projection~ : JointDiffusionProjectionContract,
) -> JointDiffusionPlan raise JointDiffusionPlanError {
  if schema_version != 1 {
    raise InvalidSchemaVersion
  }
  let workflows : ReadOnlyArray[JointDiffusionWorkflow] = ReadOnlyArray::from_array(
    workflows,
  )
  if !valid_workflows(workflows) {
    raise InvalidWorkflowSet
  }
  let stages : ReadOnlyArray[JointDiffusionStage] = ReadOnlyArray::from_array(
    stages,
  )
  if !valid_stages(workflows, stages) {
    raise InvalidStageSequence
  }
  if video_scheduler.latent() != VideoLatent ||
    audio_scheduler.latent() != AudioLatent ||
    video_scheduler.kind() != RectifiedFlowEuler ||
    audio_scheduler.kind() != RectifiedFlowEuler {
    raise InvalidScheduler
  }
  let components : ReadOnlyArray[JointDiffusionComponent] = [
    TextConditioner,
    Denoiser,
    VideoVae,
    AudioVae,
  ]
  let plan_digest = encode_plan_digest(
    schema_version, content_digest, workflows, stages, video_scheduler, audio_scheduler,
    shapes, conditioning, denoiser, projection,
  )
  {
    schema_version,
    content_digest,
    plan_digest,
    workflows,
    stages,
    components,
    video_scheduler,
    audio_scheduler,
    shapes,
    conditioning,
    denoiser,
    projection,
    packed_joint_sequence: true,
    guidance_distilled: true,
    conditioning_draws_precede_output_draws: true,
  }
}

///|
pub fn JointDiffusionPlan::schema_version(self : Self) -> Int {
  self.schema_version
}

///|
pub fn JointDiffusionPlan::content_digest(self : Self) -> @spec.ContentDigest {
  self.content_digest
}

///|
pub fn JointDiffusionPlan::plan_digest(self : Self) -> @spec.PlanDigest {
  self.plan_digest
}

///|
pub fn JointDiffusionPlan::workflows(
  self : Self,
) -> ReadOnlyArray[JointDiffusionWorkflow] {
  self.workflows
}

///|
pub fn JointDiffusionPlan::stages(
  self : Self,
) -> ReadOnlyArray[JointDiffusionStage] {
  self.stages
}

///|
pub fn JointDiffusionPlan::components(
  self : Self,
) -> ReadOnlyArray[JointDiffusionComponent] {
  self.components
}

///|
pub fn JointDiffusionPlan::video_scheduler(
  self : Self,
) -> JointDiffusionScheduler {
  self.video_scheduler
}

///|
pub fn JointDiffusionPlan::audio_scheduler(
  self : Self,
) -> JointDiffusionScheduler {
  self.audio_scheduler
}

///|
pub fn JointDiffusionPlan::shapes(self : Self) -> JointDiffusionShapeContract {
  self.shapes
}

///|
pub fn JointDiffusionPlan::conditioning(
  self : Self,
) -> JointDiffusionConditioningLimits {
  self.conditioning
}

///|
pub fn JointDiffusionPlan::denoiser(self : Self) -> JointDiffusionDenoiserShape {
  self.denoiser
}

///|
pub fn JointDiffusionPlan::projection(
  self : Self,
) -> JointDiffusionProjectionContract {
  self.projection
}

///|
pub fn JointDiffusionPlan::packed_joint_sequence(self : Self) -> Bool {
  self.packed_joint_sequence
}

///|
pub fn JointDiffusionPlan::guidance_distilled(self : Self) -> Bool {
  self.guidance_distilled
}

///|
pub fn JointDiffusionPlan::conditioning_draws_precede_output_draws(
  self : Self,
) -> Bool {
  self.conditioning_draws_precede_output_draws
}

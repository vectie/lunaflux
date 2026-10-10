///|
/// A request workflow admitted by a joint audio/video diffusion plan.
pub(all) enum JointDiffusionWorkflow {
  TextToAudioVideo
  FirstLastFrameToAudioVideo
  ReferenceToAudioVideo
} derive(Debug, Eq)

///|
/// Stable startup stage vocabulary. These stages describe component and data
/// dependencies; they are not decoder token-step operations.
pub(all) enum JointDiffusionStageKind {
  ResolveRequest
  EncodeTextConditioning
  EncodeKeyframeConditioning
  EncodeReferenceConditioning
  InitializeJointLatents
  JointAudioVideoDenoise
  DecodeVideo
  DecodeAudio
} derive(Debug, Eq)

///|
/// Whether a stage runs for every admitted request or only for a workflow
/// that supplies the corresponding conditioning input.
pub(all) enum JointDiffusionStagePresence {
  Required
  Optional
} derive(Debug, Eq)

///|
/// Weight-bearing component required by the plan.
pub(all) enum JointDiffusionComponent {
  TextConditioner
  Denoiser
  VideoVae
  AudioVae
} derive(Debug, Eq)

///|
/// Scheduler algorithm used for one latent modality.
pub(all) enum JointDiffusionSchedulerKind {
  RectifiedFlowEuler
} derive(Debug, Eq)

///|
/// Latent modality advanced by a scheduler inside the joint denoising call.
pub(all) enum JointDiffusionLatentKind {
  VideoLatent
  AudioLatent
} derive(Debug, Eq)

///|
/// Storage and conversion contract for dense latent-input projections.
pub(all) enum JointDiffusionProjectionStorage {
  Float32InputWeightBiasBFloat16Output
} derive(Debug, Eq)

///|
/// Storage and arithmetic boundary for final modality output heads.
pub(all) enum JointDiffusionOutputProjectionStorage {
  Float32InputWeightBiasFloat32Output
  BFloat16InputWeightBiasBFloat16Output
} derive(Debug, Eq)

///|
/// Packed video row order: rows are batch, frame, patch-row, patch-column;
/// vector lanes are channel, temporal offset, height offset, width offset.
pub(all) enum JointDiffusionVideoPatchLayout {
  BatchFrameHeightWidthChannelTemporalHeightWidth
} derive(Debug, Eq)

///|
/// Packed audio rows group every time row of one channel before the next.
pub(all) enum JointDiffusionAudioRowLayout {
  ChannelMajorTime
} derive(Debug, Eq)

///|
/// Payload-safe failure from generic plan construction or request-shape
/// resolution.
pub(all) suberror JointDiffusionPlanError {
  InvalidSchemaVersion
  InvalidStageSequence
  InvalidWorkflowSet
  InvalidScheduler
  InvalidShapeContract
  InvalidConditioningLimits
  InvalidCanvas
  InvalidFrameCount
  InvalidInferenceStepCount
} derive(Debug, Eq)

///|
/// One immutable pipeline stage.
pub struct JointDiffusionStage {
  priv kind : JointDiffusionStageKind
  priv presence : JointDiffusionStagePresence
} derive(Debug, Eq)

///|
/// Exact rational scheduler shift, avoiding a platform-dependent floating
/// encoding in plan identity.
pub struct JointDiffusionScheduler {
  priv latent : JointDiffusionLatentKind
  priv kind : JointDiffusionSchedulerKind
  priv shift_numerator : Int
  priv shift_denominator : Int
  priv training_timestep_count : Int
} derive(Debug, Eq)

///|
/// Bounded request conditioning envelope.
pub struct JointDiffusionConditioningLimits {
  priv max_keyframes : Int
  priv max_image_references : Int
  priv max_video_references : Int
  priv max_audio_references : Int
  priv max_total_references : Int
  priv allow_audio_only_references : Bool
} derive(Debug, Eq)

///|
/// Architecture-neutral geometry of the transformer that advances the packed
/// joint latent sequence.
pub struct JointDiffusionDenoiserShape {
  priv hidden_size : Int
  priv layer_count : Int
  priv refiner_layer_count : Int
  priv attention_head_count : Int
  priv attention_head_dimension : Int
  priv attention_rotary_dimension : Int
  priv query_key_norm_epsilon_numerator : Int
  priv query_key_norm_epsilon_denominator : Int
  priv feed_forward_dimension : Int
  priv text_conditioning_dimension : Int
  priv timestep_frequency_dimension : Int
  priv timestep_embedding_hidden_dimension : Int
  priv final_adaln_conditioning_dimension : Int
  priv final_norm_epsilon_numerator : Int
  priv final_norm_epsilon_denominator : Int
} derive(Debug, Eq)

///|
/// Family-neutral dense projection ABI owned by the canonical plan.
pub struct JointDiffusionProjectionContract {
  priv storage : JointDiffusionProjectionStorage
  priv output_storage : JointDiffusionOutputProjectionStorage
  priv video_layout : JointDiffusionVideoPatchLayout
  priv audio_layout : JointDiffusionAudioRowLayout
} derive(Debug, Eq)

///|
/// Decoder output length is independent of the nominal frame duration.
/// LatentStride retains the decoder's complete rounded latent duration.
pub(all) enum JointAudioOutputLength {
  FrameDuration
  LatentStride(Int)
} derive(Debug, Eq)

///|
/// Model and runtime geometry needed to resolve one joint audio/video request.
pub struct JointDiffusionShapeContract {
  priv frames_per_second : Int
  priv min_duration_seconds : Int
  priv max_duration_seconds : Int
  priv frames_per_chunk : Int
  priv latents_per_chunk : Int
  priv canvas_multiple : Int
  priv max_canvas_pixels : Int64
  priv min_aspect_numerator : Int
  priv min_aspect_denominator : Int
  priv max_aspect_numerator : Int
  priv max_aspect_denominator : Int
  priv video_latent_channels : Int
  priv video_spatial_compression : Int
  priv audio_latent_channels : Int
  priv audio_latent_streams : Int
  priv audio_latents_per_second : Int
  priv audio_sample_rate : Int
  priv audio_output_channels : Int
  priv audio_output_length : JointAudioOutputLength
  priv audio_vae_latent_dimension : Int
  priv audio_vae_decoder_dimension : Int
  priv min_inference_steps : Int
  priv max_inference_steps : Int
  priv video_patch_temporal : Int
  priv video_patch_height : Int
  priv video_patch_width : Int
} derive(Debug, Eq)

///|
/// Fully resolved, allocation-independent shape of one admitted request.
pub struct JointDiffusionRequestShape {
  priv height : Int
  priv width : Int
  priv frames : Int
  priv inference_steps : Int
  priv model_evaluations : Int
  priv video_latent_frames : Int
  priv video_latent_height : Int
  priv video_latent_width : Int
  priv audio_latent_frames : Int
  priv audio_token_rows : Int
  priv audio_samples : Int64
  priv audio_output_samples : Int64
} derive(Debug, Eq)

///|
/// Immutable architecture-neutral plan for a joint audio/video diffusion
/// pipeline. Family-specific specifications build this type but do not leak
/// through it.
pub struct JointDiffusionPlan {
  priv schema_version : Int
  priv content_digest : @spec.ContentDigest
  priv plan_digest : @spec.PlanDigest
  priv workflows : ReadOnlyArray[JointDiffusionWorkflow]
  priv stages : ReadOnlyArray[JointDiffusionStage]
  priv components : ReadOnlyArray[JointDiffusionComponent]
  priv video_scheduler : JointDiffusionScheduler
  priv audio_scheduler : JointDiffusionScheduler
  priv shapes : JointDiffusionShapeContract
  priv conditioning : JointDiffusionConditioningLimits
  priv denoiser : JointDiffusionDenoiserShape
  priv projection : JointDiffusionProjectionContract
  priv packed_joint_sequence : Bool
  priv guidance_distilled : Bool
  priv conditioning_draws_precede_output_draws : Bool
} derive(Debug, Eq)

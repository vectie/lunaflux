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
  priv shift_numerator : Int32
  priv shift_denominator : Int32
  priv training_timestep_count : Int32
} derive(Debug, Eq)

///|
/// Bounded request conditioning envelope.
pub struct JointDiffusionConditioningLimits {
  priv max_keyframes : Int32
  priv max_image_references : Int32
  priv max_video_references : Int32
  priv max_audio_references : Int32
  priv max_total_references : Int32
  priv allow_audio_only_references : Bool
} derive(Debug, Eq)

///|
/// Architecture-neutral geometry of the transformer that advances the packed
/// joint latent sequence.
pub struct JointDiffusionDenoiserShape {
  priv hidden_size : Int32
  priv layer_count : Int32
  priv refiner_layer_count : Int32
  priv attention_head_count : Int32
  priv attention_head_dimension : Int32
  priv attention_rotary_dimension : Int32
  priv query_key_norm_epsilon_numerator : Int32
  priv query_key_norm_epsilon_denominator : Int32
  priv feed_forward_dimension : Int32
  priv text_conditioning_dimension : Int32
  priv timestep_frequency_dimension : Int32
  priv timestep_embedding_hidden_dimension : Int32
  priv final_adaln_conditioning_dimension : Int32
  priv final_norm_epsilon_numerator : Int32
  priv final_norm_epsilon_denominator : Int32
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
  LatentStride(Int32)
} derive(Debug, Eq)

///|
/// Requested step cardinality and scalar arithmetic are model semantics, not
/// device scheduling. InclusiveZeroFloat32 counts sigma points (N-1 updates).
pub(all) enum JointDiffusionSigmaGrid {
  TrainingFloorFloat64
  InclusiveZeroFloat32
} derive(Debug, Eq)

///|
/// Model and runtime geometry needed to resolve one joint audio/video request.
pub struct JointDiffusionShapeContract {
  priv frames_per_second : Int32
  priv min_duration_seconds : Int32
  priv max_duration_seconds : Int32
  priv frames_per_chunk : Int32
  priv latents_per_chunk : Int32
  priv canvas_multiple : Int32
  priv max_canvas_pixels : Int64
  priv min_aspect_numerator : Int32
  priv min_aspect_denominator : Int32
  priv max_aspect_numerator : Int32
  priv max_aspect_denominator : Int32
  priv video_latent_channels : Int32
  priv video_spatial_compression : Int32
  priv audio_latent_channels : Int32
  priv audio_latent_streams : Int32
  priv audio_latents_per_second : Int32
  priv audio_sample_rate : Int32
  priv audio_output_channels : Int32
  priv audio_output_length : JointAudioOutputLength
  priv audio_vae_latent_dimension : Int32
  priv audio_vae_decoder_dimension : Int32
  priv min_inference_steps : Int32
  priv max_inference_steps : Int32
  priv sigma_grid : JointDiffusionSigmaGrid
  priv video_patch_temporal : Int32
  priv video_patch_height : Int32
  priv video_patch_width : Int32
} derive(Debug, Eq)

///|
/// Fully resolved, allocation-independent shape of one admitted request.
pub struct JointDiffusionRequestShape {
  priv height : Int32
  priv width : Int32
  priv frames : Int32
  priv inference_steps : Int32
  priv model_evaluations : Int32
  priv sigma_grid : JointDiffusionSigmaGrid
  priv video_latent_frames : Int32
  priv video_latent_height : Int32
  priv video_latent_width : Int32
  priv audio_latent_frames : Int32
  priv audio_token_rows : Int32
  priv audio_samples : Int64
  priv audio_output_samples : Int64
} derive(Debug, Eq)

///|
/// Immutable architecture-neutral plan for a joint audio/video diffusion
/// pipeline. Family-specific specifications build this type but do not leak
/// through it.
pub struct JointDiffusionPlan {
  priv schema_version : Int32
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

///|
pub extend JointDiffusionWorkflow with @debug.Debug::{to_repr}

///|
pub extend JointDiffusionWorkflow with Eq::{not_equal, equal}

///|
pub extend JointDiffusionStageKind with @debug.Debug::{to_repr}

///|
pub extend JointDiffusionStageKind with Eq::{not_equal, equal}

///|
pub extend JointDiffusionStagePresence with @debug.Debug::{to_repr}

///|
pub extend JointDiffusionStagePresence with Eq::{not_equal, equal}

///|
pub extend JointDiffusionComponent with @debug.Debug::{to_repr}

///|
pub extend JointDiffusionComponent with Eq::{not_equal, equal}

///|
pub extend JointDiffusionSchedulerKind with @debug.Debug::{to_repr}

///|
pub extend JointDiffusionSchedulerKind with Eq::{not_equal, equal}

///|
pub extend JointDiffusionLatentKind with @debug.Debug::{to_repr}

///|
pub extend JointDiffusionLatentKind with Eq::{not_equal, equal}

///|
pub extend JointDiffusionProjectionStorage with @debug.Debug::{to_repr}

///|
pub extend JointDiffusionProjectionStorage with Eq::{not_equal, equal}

///|
pub extend JointDiffusionOutputProjectionStorage with @debug.Debug::{to_repr}

///|
pub extend JointDiffusionOutputProjectionStorage with Eq::{not_equal, equal}

///|
pub extend JointDiffusionVideoPatchLayout with @debug.Debug::{to_repr}

///|
pub extend JointDiffusionVideoPatchLayout with Eq::{not_equal, equal}

///|
pub extend JointDiffusionAudioRowLayout with @debug.Debug::{to_repr}

///|
pub extend JointDiffusionAudioRowLayout with Eq::{not_equal, equal}

///|
pub extend JointDiffusionPlanError with @debug.Debug::{to_repr}

///|
pub extend JointDiffusionPlanError with Eq::{not_equal, equal}

///|
pub extend JointDiffusionStage with @debug.Debug::{to_repr}

///|
pub extend JointDiffusionStage with Eq::{not_equal, equal}

///|
pub extend JointDiffusionScheduler with @debug.Debug::{to_repr}

///|
pub extend JointDiffusionScheduler with Eq::{not_equal, equal}

///|
pub extend JointDiffusionConditioningLimits with @debug.Debug::{to_repr}

///|
pub extend JointDiffusionConditioningLimits with Eq::{not_equal, equal}

///|
pub extend JointDiffusionDenoiserShape with @debug.Debug::{to_repr}

///|
pub extend JointDiffusionDenoiserShape with Eq::{not_equal, equal}

///|
pub extend JointDiffusionProjectionContract with @debug.Debug::{to_repr}

///|
pub extend JointDiffusionProjectionContract with Eq::{not_equal, equal}

///|
pub extend JointAudioOutputLength with @debug.Debug::{to_repr}

///|
pub extend JointAudioOutputLength with Eq::{not_equal, equal}

///|
pub extend JointDiffusionSigmaGrid with @debug.Debug::{to_repr}

///|
pub extend JointDiffusionSigmaGrid with Eq::{not_equal, equal}

///|
pub extend JointDiffusionShapeContract with @debug.Debug::{to_repr}

///|
pub extend JointDiffusionShapeContract with Eq::{not_equal, equal}

///|
pub extend JointDiffusionRequestShape with @debug.Debug::{to_repr}

///|
pub extend JointDiffusionRequestShape with Eq::{not_equal, equal}

///|
pub extend JointDiffusionPlan with @debug.Debug::{to_repr}

///|
pub extend JointDiffusionPlan with Eq::{not_equal, equal}

///|
/// Payload-safe rejection at the immutable joint diffusion I/O boundary.
pub(all) suberror JointDiffusionIoPlanError {
  WorkflowNotAdmitted
  InvalidTextConditioning
  InvalidMediaDescriptor
  InvalidKeyframeConditioning
  InvalidReferenceConditioning
  InvalidOutputShape
  InvalidGuidancePolicy
  InvalidResultMetadata
  InvalidCancellationProgress
  InvalidPackedTimestepSelection
  InvalidPackedModalitySelection
  InvalidDistinctTimestepValues
} derive(Debug, Eq)

///|
/// Text conditioning that has already been validated and tokenized upstream.
/// The digest identifies the exact normalized token payload; this package does
/// not own a tokenizer or raw prompt text.
pub struct ValidatedTextConditioning {
  priv content_digest : @spec.ContentDigest
  priv token_count : Int
} derive(Debug, Eq)

///|
/// An image descriptor whose decoded pixels have already been validated.
pub struct ValidatedImageInput {
  priv content_digest : @spec.ContentDigest
  priv height : Int
  priv width : Int
} derive(Debug, Eq)

///|
/// A video descriptor whose decoded frames have already been validated.
pub struct ValidatedVideoInput {
  priv content_digest : @spec.ContentDigest
  priv height : Int
  priv width : Int
  priv frame_count : Int
  priv frames_per_second : Int
} derive(Debug, Eq)

///|
/// An audio descriptor whose decoded samples have already been validated.
pub struct ValidatedAudioInput {
  priv content_digest : @spec.ContentDigest
  priv sample_count : Int64
  priv sample_rate : Int
  priv channel_count : Int
} derive(Debug, Eq)

///|
/// Semantic position of a first/last-frame conditioning image.
pub(all) enum JointKeyframePosition {
  FirstFrame
  LastFrame
} derive(Debug, Eq)

///|
/// One positioned keyframe. Normalization canonicalizes first before last and
/// rejects duplicate positions.
pub struct JointKeyframeConditioning {
  priv position : JointKeyframePosition
  priv image : ValidatedImageInput
} derive(Debug, Eq)

///|
/// One ordered Ref2VA conditioning item. Array order is semantic and is bound
/// into request identity.
pub(all) enum JointReferenceConditioning {
  ImageReference(ValidatedImageInput)
  VideoReference(ValidatedVideoInput)
  AudioReference(ValidatedAudioInput)
} derive(Debug, Eq)

///|
/// Guidance policy admitted by the generic joint diffusion plan. Schema v1
/// plans are guidance-distilled and therefore admit exactly unit guidance.
pub(all) enum JointGuidancePolicy {
  DistilledUnitGuidance
} derive(Debug, Eq)

///|
/// Cancellation is observable only at stage boundaries. This is metadata, not
/// a scheduler or cancellation-token implementation.
pub(all) enum JointCancellationPolicy {
  StageBoundaryCancellation
} derive(Debug, Eq)

///|
/// Typed terminal cancellation reasons carried without transport payloads.
pub(all) enum JointCancellationReason {
  CallerRequested
  DeadlineExceeded
  RuntimeShutdown
} derive(Debug, Eq)

///|
/// Terminal metadata status. It does not contain encoded or decoded media.
pub(all) enum JointDiffusionResultStatus {
  Succeeded
  Cancelled(JointCancellationReason)
} derive(Debug, Eq)

///|
/// Immutable request-to-execution boundary for joint audio/video diffusion.
pub struct JointDiffusionIoPlan {
  priv schema_version : Int
  priv identity : @spec.ModelIdentity
  priv request_digest : @spec.PlanDigest
  priv workflow : @joint_diffusion_plan.JointDiffusionWorkflow
  priv text : ValidatedTextConditioning
  priv keyframes : ReadOnlyArray[JointKeyframeConditioning]
  priv references : ReadOnlyArray[JointReferenceConditioning]
  priv shape : @joint_diffusion_plan.JointDiffusionRequestShape
  priv video_latent_channels : Int
  priv video_patch_temporal : Int
  priv video_patch_height : Int
  priv video_patch_width : Int
  priv audio_latent_channels : Int
  priv audio_sample_rate : Int
  priv audio_output_channels : Int
  priv seed : Int64
  priv guidance : JointGuidancePolicy
  priv cancellation : JointCancellationPolicy
  priv stage_count : Int
} derive(Debug, Eq)

///|
/// Digest-bound terminal result metadata. Exact output counts are present only
/// for success; cancellation records bounded completed-stage progress.
pub struct JointDiffusionResultMetadata {
  priv identity : @spec.ModelIdentity
  priv request_digest : @spec.PlanDigest
  priv status : JointDiffusionResultStatus
  priv completed_stage_count : Int
  priv video_frame_count : Int?
  priv audio_sample_count : Int64?
} derive(Debug, Eq)

///|
pub struct JointPackedTimestepSelection {
  priv identity : @spec.ModelIdentity
  priv request_digest : @spec.PlanDigest
  priv digest : @spec.PlanDigest
  priv evaluation_index : Int
  priv packed_row_count : Int
  priv distinct_timestep_count : Int
  priv indices : ReadOnlyArray[Int]
} derive(Debug, Eq)

///|
pub struct JointPackedModalitySelection {
  priv identity : @spec.ModelIdentity
  priv request_digest : @spec.PlanDigest
  priv packed_timestep_digest : @spec.PlanDigest
  priv digest : @spec.PlanDigest
  priv packed_row_count : Int
  priv video_indices : ReadOnlyArray[Int]
  priv audio_indices : ReadOnlyArray[Int]
} derive(Debug, Eq)

///|
pub struct JointDistinctTimestepValues {
  priv identity : @spec.ModelIdentity
  priv request_digest : @spec.PlanDigest
  priv packed_timestep_digest : @spec.PlanDigest
  priv digest : @spec.PlanDigest
  priv values : ReadOnlyArray[Float]
}

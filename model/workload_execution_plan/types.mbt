///|
/// Architecture-neutral execution topology selected once at engine startup.
/// This is deliberately a workload distinction rather than model-family
/// identity.
pub(all) enum WorkloadExecutionKind {
  DecoderOnlyText
  MultimodalConditionalGeneration
  JointAudioVideoDiffusion
} derive(Debug, Eq)

///|
/// Invalid architecture-neutral workload admission envelope.
pub(all) suberror WorkloadExecutionPlanError {
  InvalidSchemaVersion
  InvalidDecoderContract
  InvalidDiffusionContract
} derive(Debug, Eq)

///|
/// Immutable architecture-neutral projection of one authenticated semantic
/// plan. The concrete plan remains owned by the prepared execution owner; core
/// lifecycle code sees only this bounded admission envelope.
pub struct WorkloadExecutionPlan {
  priv kind : WorkloadExecutionKind
  priv identity : @spec.ModelIdentity
  priv schema_version : Int
  priv decoder_layer_count : Int?
  priv maximum_token_context : Int?
} derive(Debug, Eq)

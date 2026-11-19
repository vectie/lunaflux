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
  priv schema_version : Int32
  priv decoder_layer_count : Int32?
  priv maximum_token_context : Int32?
} derive(Debug, Eq)

///|
pub extend WorkloadExecutionKind with @debug.Debug::{to_repr}

///|
pub extend WorkloadExecutionKind with Eq::{not_equal, equal}

///|
pub extend WorkloadExecutionPlanError with @debug.Debug::{to_repr}

///|
pub extend WorkloadExecutionPlanError with Eq::{not_equal, equal}

///|
pub extend WorkloadExecutionPlan with @debug.Debug::{to_repr}

///|
pub extend WorkloadExecutionPlan with Eq::{not_equal, equal}

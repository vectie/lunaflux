///|
pub extend JointDiffusionAudioVaeConvPreStageKind with @debug.Debug::{to_repr}

///|
pub extend JointDiffusionAudioVaeConvPreStageKind with Eq::{not_equal, equal}

///|
pub extend JointDiffusionAudioVaeFirstAmpStageKind with @debug.Debug::{to_repr}

///|
pub extend JointDiffusionAudioVaeFirstAmpStageKind with Eq::{not_equal, equal}

///|
pub extend JointDiffusionAudioVaeFirstUpsampleStageKind with @debug.Debug::{
  to_repr,
}

///|
pub extend JointDiffusionAudioVaeFirstUpsampleStageKind with Eq::{
  not_equal,
  equal,
}

///|
pub extend JointDiffusionCudaNumericalContract with @debug.Debug::{to_repr}

///|
pub extend JointDiffusionCudaNumericalContract with Eq::{not_equal, equal}

///|
pub extend JointDiffusionCudaAotInvalidKind with @debug.Debug::{to_repr}

///|
pub extend JointDiffusionCudaAotInvalidKind with Eq::{not_equal, equal}

///|
pub extend JointDiffusionCudaAotError with @debug.Debug::{to_repr}

///|
pub extend JointDiffusionCudaAotError with Eq::{not_equal, equal}

///|
pub extend JointDiffusionCudaSourceDigest with @debug.Debug::{to_repr}

///|
pub extend JointDiffusionCudaSourceDigest with Eq::{not_equal, equal}

///|
pub extend JointDiffusionCudaSourceDigest with Hash::{hash, hash_combine}

///|
pub extend JointDiffusionCudaRecipeDigest with @debug.Debug::{to_repr}

///|
pub extend JointDiffusionCudaRecipeDigest with Eq::{not_equal, equal}

///|
pub extend JointDiffusionCudaRecipeDigest with Hash::{hash, hash_combine}

///|
pub extend JointDiffusionAttentionRowScope with @debug.Debug::{to_repr}

///|
pub extend JointDiffusionAttentionRowScope with Eq::{not_equal, equal}

///|
pub extend JointDiffusionAttentionStageKind with @debug.Debug::{to_repr}

///|
pub extend JointDiffusionAttentionStageKind with Eq::{not_equal, equal}

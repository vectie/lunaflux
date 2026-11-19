///|
pub extend AdvancedDecoderOperandRegionPlanStatus with @debug.Debug::{to_repr}

///|
pub extend AdvancedDecoderOperandRegionPlanStatus with Eq::{not_equal, equal}

///|
pub extend AdvancedDecoderOperandRegionPlanDigest with @debug.Debug::{to_repr}

///|
pub extend AdvancedDecoderOperandRegionPlanDigest with Eq::{not_equal, equal}

///|
pub extend AdvancedDecoderOperandRegionPlanDigest with Hash::{
  hash,
  hash_combine,
}

///|
pub extend AdvancedDecoderOperandRegionPlanError with @debug.Debug::{to_repr}

///|
pub extend AdvancedDecoderOperandRegionPlanError with Eq::{not_equal, equal}

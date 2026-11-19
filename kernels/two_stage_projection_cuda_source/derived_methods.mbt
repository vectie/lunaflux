///|
pub extend TwoStageProjectionOperation with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend TwoStageProjectionOperation with Eq::{not_equal, equal}

///|
pub extend TwoStageProjectionNumericalContract with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend TwoStageProjectionNumericalContract with Eq::{not_equal, equal}

///|
pub extend TwoStageProjectionCudaSourceError with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend TwoStageProjectionCudaSourceError with Eq::{not_equal, equal}

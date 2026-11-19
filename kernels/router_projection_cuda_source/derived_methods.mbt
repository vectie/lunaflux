///|
pub extend RouterProjectionCudaOperation with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend RouterProjectionCudaOperation with Eq::{not_equal, equal}

///|
pub extend RouterProjectionNumericalContract with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend RouterProjectionNumericalContract with Eq::{not_equal, equal}

///|
pub extend RouterProjectionCudaSourceError with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend RouterProjectionCudaSourceError with Eq::{not_equal, equal}

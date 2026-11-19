///|
pub extend RecurrentAttentionProjectionOperation with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend RecurrentAttentionProjectionOperation with Eq::{not_equal, equal}

///|
pub extend RecurrentAttentionProjectionNumericalContract with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend RecurrentAttentionProjectionNumericalContract with Eq::{
  not_equal,
  equal,
}

///|
pub extend RecurrentAttentionProjectionCudaSourceError with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend RecurrentAttentionProjectionCudaSourceError with Eq::{
  not_equal,
  equal,
}

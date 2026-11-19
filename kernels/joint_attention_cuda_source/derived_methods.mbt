///|
pub extend JointAttentionNumericalContract with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend JointAttentionNumericalContract with Eq::{not_equal, equal}

///|
pub extend JointAttentionCudaSourceError with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend JointAttentionCudaSourceError with Eq::{not_equal, equal}

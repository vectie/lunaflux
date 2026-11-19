///|
pub extend JointAudioAmpCudaNumericalContract with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend JointAudioAmpCudaNumericalContract with Eq::{not_equal, equal}

///|
pub extend JointAudioAmpCudaSourceError with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend JointAudioAmpCudaSourceError with Eq::{not_equal, equal}

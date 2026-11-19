///|
pub extend JointAudioAmpParameterRole with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend JointAudioAmpParameterRole with Eq::{not_equal, equal}

///|
pub extend JointAudioAmpParameterLayoutError with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend JointAudioAmpParameterLayoutError with Eq::{not_equal, equal}

///|
pub extend JointAudioAmpParameterRegion with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend JointAudioAmpParameterRegion with Eq::{not_equal, equal}

///|
pub extend JointAudioAmpParameterLayoutDigest with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend JointAudioAmpParameterLayoutDigest with Eq::{not_equal, equal}

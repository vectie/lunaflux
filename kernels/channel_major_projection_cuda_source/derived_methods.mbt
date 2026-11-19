///|
pub extend ChannelMajorProjectionOperation with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend ChannelMajorProjectionOperation with Eq::{not_equal, equal}

///|
pub extend ChannelMajorProjectionNumericalContract with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend ChannelMajorProjectionNumericalContract with Eq::{not_equal, equal}

///|
pub extend ChannelMajorProjectionCudaSourceError with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend ChannelMajorProjectionCudaSourceError with Eq::{not_equal, equal}

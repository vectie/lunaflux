///|
pub extend HyperConnectionMixCudaSourceError with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend HyperConnectionMixCudaSourceError with Eq::{not_equal, equal}

///|
pub extend HyperConnectionMixNumericalContract with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend HyperConnectionMixNumericalContract with Eq::{not_equal, equal}

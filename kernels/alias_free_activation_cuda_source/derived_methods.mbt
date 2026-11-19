///|
pub extend AliasFreeActivationNumericalContract with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend AliasFreeActivationNumericalContract with Eq::{not_equal, equal}

///|
pub extend AliasFreeActivationCudaSourceError with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend AliasFreeActivationCudaSourceError with Eq::{not_equal, equal}

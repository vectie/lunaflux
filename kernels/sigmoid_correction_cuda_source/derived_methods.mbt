///|
pub extend SigmoidCorrectionCudaOperation with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend SigmoidCorrectionCudaOperation with Eq::{not_equal, equal}

///|
pub extend SigmoidCorrectionNumericalContract with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend SigmoidCorrectionNumericalContract with Eq::{not_equal, equal}

///|
pub extend SigmoidCorrectionCudaSourceError with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend SigmoidCorrectionCudaSourceError with Eq::{not_equal, equal}

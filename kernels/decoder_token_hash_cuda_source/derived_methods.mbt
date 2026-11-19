///|
pub extend DecoderTokenHashCudaOperation with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend DecoderTokenHashCudaOperation with Eq::{not_equal, equal}

///|
pub extend DecoderTokenHashNumericalContract with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend DecoderTokenHashNumericalContract with Eq::{not_equal, equal}

///|
pub extend DecoderTokenHashCudaSourceError with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend DecoderTokenHashCudaSourceError with Eq::{not_equal, equal}

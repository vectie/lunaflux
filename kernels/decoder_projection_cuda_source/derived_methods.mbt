///|
pub extend DecoderProjectionCudaOperation with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend DecoderProjectionCudaOperation with Eq::{not_equal, equal}

///|
pub extend DecoderProjectionNumericalContract with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend DecoderProjectionNumericalContract with Eq::{not_equal, equal}

///|
pub extend DecoderProjectionCudaSourceError with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend DecoderProjectionCudaSourceError with Eq::{not_equal, equal}

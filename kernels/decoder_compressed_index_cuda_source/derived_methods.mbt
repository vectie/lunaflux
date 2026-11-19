///|
pub extend DecoderCompressedIndexCudaOperation with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend DecoderCompressedIndexCudaOperation with Eq::{not_equal, equal}

///|
pub extend DecoderCompressedIndexNumericalContract with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend DecoderCompressedIndexNumericalContract with Eq::{not_equal, equal}

///|
pub extend DecoderCompressedIndexCudaSourceError with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend DecoderCompressedIndexCudaSourceError with Eq::{not_equal, equal}

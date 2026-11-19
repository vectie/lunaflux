///|
pub extend DecoderWindowCompressedJoinCudaOperation with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend DecoderWindowCompressedJoinCudaOperation with Eq::{not_equal, equal}

///|
pub extend DecoderWindowCompressedJoinNumericalContract with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend DecoderWindowCompressedJoinNumericalContract with Eq::{
  not_equal,
  equal,
}

///|
pub extend DecoderWindowCompressedJoinCudaSourceError with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend DecoderWindowCompressedJoinCudaSourceError with Eq::{
  not_equal,
  equal,
}

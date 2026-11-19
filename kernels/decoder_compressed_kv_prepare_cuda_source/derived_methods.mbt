///|
pub extend DecoderCompressedKvPrepareCudaSourceError with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend DecoderCompressedKvPrepareCudaSourceError with Eq::{not_equal, equal}

///|
pub extend DecoderCompressedKvPrepareOperation with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend DecoderCompressedKvPrepareOperation with Eq::{not_equal, equal}

///|
pub extend DecoderCompressedKvPrepareNumericalContract with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend DecoderCompressedKvPrepareNumericalContract with Eq::{
  not_equal,
  equal,
}

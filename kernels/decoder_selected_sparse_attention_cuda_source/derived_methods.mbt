///|
pub extend DecoderSelectedSparseAttentionCudaOperation with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend DecoderSelectedSparseAttentionCudaOperation with Eq::{
  not_equal,
  equal,
}

///|
pub extend DecoderSelectedSparseAttentionNumericalContract with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend DecoderSelectedSparseAttentionNumericalContract with Eq::{
  not_equal,
  equal,
}

///|
pub extend DecoderSelectedSparseAttentionCudaSourceError with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend DecoderSelectedSparseAttentionCudaSourceError with Eq::{
  not_equal,
  equal,
}

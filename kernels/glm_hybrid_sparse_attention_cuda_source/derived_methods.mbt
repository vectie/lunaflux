///|
pub extend GlmHybridSparseAttentionCudaOperation with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend GlmHybridSparseAttentionCudaOperation with Eq::{not_equal, equal}

///|
pub extend GlmHybridSparseAttentionNumericalContract with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend GlmHybridSparseAttentionNumericalContract with Eq::{not_equal, equal}

///|
pub extend GlmHybridSparseAttentionCudaSourceError with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend GlmHybridSparseAttentionCudaSourceError with Eq::{not_equal, equal}

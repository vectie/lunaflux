///|
pub extend GlmHybridDsaIndexCudaOperation with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend GlmHybridDsaIndexCudaOperation with Eq::{not_equal, equal}

///|
pub extend GlmHybridDsaIndexNumericalContract with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend GlmHybridDsaIndexNumericalContract with Eq::{not_equal, equal}

///|
pub extend GlmHybridDsaIndexCudaSourceError with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend GlmHybridDsaIndexCudaSourceError with Eq::{not_equal, equal}

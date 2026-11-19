///|
pub extend GlmHybridDsaPoolingCudaOperation with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend GlmHybridDsaPoolingCudaOperation with Eq::{not_equal, equal}

///|
pub extend GlmHybridDsaPoolingNumericalContract with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend GlmHybridDsaPoolingNumericalContract with Eq::{not_equal, equal}

///|
pub extend GlmHybridDsaPoolingCudaSourceError with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend GlmHybridDsaPoolingCudaSourceError with Eq::{not_equal, equal}

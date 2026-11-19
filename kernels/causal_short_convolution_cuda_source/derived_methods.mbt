///|
pub extend CausalShortConvolutionOperation with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend CausalShortConvolutionOperation with Eq::{not_equal, equal}

///|
pub extend CausalShortConvolutionNumericalContract with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend CausalShortConvolutionNumericalContract with Eq::{not_equal, equal}

///|
pub extend CausalShortConvolutionCudaSourceError with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend CausalShortConvolutionCudaSourceError with Eq::{not_equal, equal}

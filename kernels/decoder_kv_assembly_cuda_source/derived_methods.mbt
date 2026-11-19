///|
pub extend DecoderKvAssemblyCudaSourceError with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend DecoderKvAssemblyCudaSourceError with Eq::{not_equal, equal}

///|
pub extend DecoderKvAssemblyNumericalContract with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend DecoderKvAssemblyNumericalContract with Eq::{not_equal, equal}

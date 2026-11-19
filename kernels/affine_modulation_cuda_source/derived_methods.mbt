///|
pub extend AffineModulationCudaOperation with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend AffineModulationCudaOperation with Eq::{not_equal, equal}

///|
pub extend AffineModulationNumericalContract with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend AffineModulationNumericalContract with Eq::{not_equal, equal}

///|
pub extend AffineModulationCudaSourceError with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend AffineModulationCudaSourceError with Eq::{not_equal, equal}

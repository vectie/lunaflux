///|
pub extend RecurrentDeltaOperation with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend RecurrentDeltaOperation with Eq::{not_equal, equal}

///|
pub extend RecurrentDeltaNumericalContract with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend RecurrentDeltaNumericalContract with Eq::{not_equal, equal}

///|
pub extend RecurrentDeltaCudaSourceError with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend RecurrentDeltaCudaSourceError with Eq::{not_equal, equal}

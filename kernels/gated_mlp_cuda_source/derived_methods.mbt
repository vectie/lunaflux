///|
pub extend GatedMlpNumericalContract with @debug.Debug::{to_repr}

///|
pub extend GatedMlpNumericalContract with Eq::{not_equal, equal}

///|
pub extend GatedMlpCudaSourceError with @debug.Debug::{to_repr}

///|
pub extend GatedMlpCudaSourceError with Eq::{not_equal, equal}

///|
pub extend GatedRmsNormCudaOperation with @debug.Debug::{to_repr}

///|
pub extend GatedRmsNormCudaOperation with Eq::{not_equal, equal}

///|
pub extend GatedRmsNormNumericalContract with @debug.Debug::{to_repr}

///|
pub extend GatedRmsNormNumericalContract with Eq::{not_equal, equal}

///|
pub extend GatedRmsNormCudaSourceError with @debug.Debug::{to_repr}

///|
pub extend GatedRmsNormCudaSourceError with Eq::{not_equal, equal}

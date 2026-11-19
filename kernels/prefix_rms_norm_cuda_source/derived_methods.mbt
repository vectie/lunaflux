///|
pub extend PrefixRmsNormCudaOperation with @debug.Debug::{to_repr}

///|
pub extend PrefixRmsNormCudaOperation with Eq::{not_equal, equal}

///|
pub extend PrefixRmsNormNumericalContract with @debug.Debug::{to_repr}

///|
pub extend PrefixRmsNormNumericalContract with Eq::{not_equal, equal}

///|
pub extend PrefixRmsNormCudaSourceError with @debug.Debug::{to_repr}

///|
pub extend PrefixRmsNormCudaSourceError with Eq::{not_equal, equal}

///|
pub extend DecayControlCudaOperation with @debug.Debug::{to_repr}

///|
pub extend DecayControlCudaOperation with Eq::{not_equal, equal}

///|
pub extend DecayControlNumericalContract with @debug.Debug::{to_repr}

///|
pub extend DecayControlNumericalContract with Eq::{not_equal, equal}

///|
pub extend DecayControlCudaSourceError with @debug.Debug::{to_repr}

///|
pub extend DecayControlCudaSourceError with Eq::{not_equal, equal}

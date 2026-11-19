///|
pub extend Fp8ReleaseWorkerRule with @debug.Debug::{to_repr}

///|
pub extend Fp8ReleaseWorkerRule with Eq::{not_equal, equal}

///|
pub extend Fp8ReleaseWorkerError with @debug.Debug::{to_repr}

///|
pub extend Fp8ReleaseWorkerError with Eq::{not_equal, equal}

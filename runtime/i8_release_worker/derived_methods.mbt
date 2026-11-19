///|
pub extend I8ReleaseWorkerRule with @debug.Debug::{to_repr}

///|
pub extend I8ReleaseWorkerRule with Eq::{not_equal, equal}

///|
pub extend I8ReleaseWorkerError with @debug.Debug::{to_repr}

///|
pub extend I8ReleaseWorkerError with Eq::{not_equal, equal}

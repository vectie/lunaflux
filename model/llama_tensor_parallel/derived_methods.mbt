///|
pub extend LlamaTensorParallelDimension with @debug.Debug::{to_repr}

///|
pub extend LlamaTensorParallelDimension with Eq::{not_equal, equal}

///|
pub extend LlamaTensorParallelError with @debug.Debug::{to_repr}

///|
pub extend LlamaTensorParallelError with Eq::{not_equal, equal}

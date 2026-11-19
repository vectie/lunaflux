///|
pub extend TensorParallelCollectiveStage with @debug.Debug::{to_repr}

///|
pub extend TensorParallelCollectiveStage with Eq::{not_equal, equal}

///|
pub extend TensorParallelCollectiveError with @debug.Debug::{to_repr}

///|
pub extend TensorParallelCollectiveError with Eq::{not_equal, equal}

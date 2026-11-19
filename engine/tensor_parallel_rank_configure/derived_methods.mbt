///|
pub extend TensorParallelRankConfigureRule with @debug.Debug::{to_repr}

///|
pub extend TensorParallelRankConfigureRule with Eq::{not_equal, equal}

///|
pub extend TensorParallelRankConfigureError with @debug.Debug::{to_repr}

///|
pub extend TensorParallelRankConfigureError with Eq::{not_equal, equal}

///|
pub extend TensorParallelRankConfigureDigest with @debug.Debug::{to_repr}

///|
pub extend TensorParallelRankConfigureDigest with Eq::{not_equal, equal}

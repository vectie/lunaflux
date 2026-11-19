///|
pub extend TensorParallelRankChildRule with @debug.Debug::{to_repr}

///|
pub extend TensorParallelRankChildRule with Eq::{not_equal, equal}

///|
pub extend TensorParallelRankChildError with @debug.Debug::{to_repr}

///|
pub extend TensorParallelRankChildError with Eq::{not_equal, equal}

///|
pub extend TensorParallelRankChildReadyProgress with @debug.Debug::{to_repr}

///|
pub extend TensorParallelRankChildReadyProgress with Eq::{not_equal, equal}

///|
pub extend TensorParallelRankChildControlProgress with @debug.Debug::{to_repr}

///|
pub extend TensorParallelRankChildControlProgress with Eq::{not_equal, equal}

///|
pub extend TensorParallelDevicePlanRule with @debug.Debug::{to_repr}

///|
pub extend TensorParallelDevicePlanRule with Eq::{not_equal, equal}

///|
pub extend TensorParallelDevicePlanError with @debug.Debug::{to_repr}

///|
pub extend TensorParallelDevicePlanError with Eq::{not_equal, equal}

///|
pub extend TensorParallelRankDevice with @debug.Debug::{to_repr}

///|
pub extend TensorParallelRankDevice with Eq::{not_equal, equal}

///|
pub extend TensorParallelMatrixView with @debug.Debug::{to_repr}

///|
pub extend TensorParallelMatrixView with Eq::{not_equal, equal}

///|
pub extend TensorParallelDevicePlan with @debug.Debug::{to_repr}

///|
pub extend TensorParallelDevicePlan with Eq::{not_equal, equal}

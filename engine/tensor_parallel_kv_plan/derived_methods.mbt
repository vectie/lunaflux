///|
pub extend TensorParallelKvPlanRule with @debug.Debug::{to_repr}

///|
pub extend TensorParallelKvPlanRule with Eq::{not_equal, equal}

///|
pub extend TensorParallelKvPlanError with @debug.Debug::{to_repr}

///|
pub extend TensorParallelKvPlanError with Eq::{not_equal, equal}

///|
pub extend TensorParallelKvPlan with @debug.Debug::{to_repr}

///|
pub extend TensorParallelKvPlan with Eq::{not_equal, equal}

///|
pub extend TensorParallelKvContractDigest with @debug.Debug::{to_repr}

///|
pub extend TensorParallelKvContractDigest with Eq::{not_equal, equal}

///|
pub extend TensorParallelKvRankBinding with @debug.Debug::{to_repr}

///|
pub extend TensorParallelKvRankBinding with Eq::{not_equal, equal}

///|
pub extend TensorParallelKvPageRegion with @debug.Debug::{to_repr}

///|
pub extend TensorParallelKvPageRegion with Eq::{not_equal, equal}

///|
pub extend RankDeviceFacts with @debug.Debug::{to_repr}

///|
pub extend RankDeviceFacts with Eq::{not_equal, equal}

///|
pub extend RankRuntimeFacts with @debug.Debug::{to_repr}

///|
pub extend RankRuntimeFacts with Eq::{not_equal, equal}

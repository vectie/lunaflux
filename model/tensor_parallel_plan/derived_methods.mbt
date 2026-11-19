///|
pub extend TensorParallelPlanRule with @debug.Debug::{to_repr}

///|
pub extend TensorParallelPlanRule with Eq::{not_equal, equal}

///|
pub extend TensorParallelPlanError with @debug.Debug::{to_repr}

///|
pub extend TensorParallelPlanError with Eq::{not_equal, equal}

///|
pub extend TensorParallelTopology with @debug.Debug::{to_repr}

///|
pub extend TensorParallelTopology with Eq::{not_equal, equal}

///|
pub extend TensorPlacement with @debug.Debug::{to_repr}

///|
pub extend TensorPlacement with Eq::{not_equal, equal}

///|
pub extend TensorShardExtent with @debug.Debug::{to_repr}

///|
pub extend TensorShardExtent with Eq::{not_equal, equal}

///|
pub extend TensorArenaRegion with @debug.Debug::{to_repr}

///|
pub extend TensorArenaRegion with Eq::{not_equal, equal}

///|
pub extend TensorTransferSegment with @debug.Debug::{to_repr}

///|
pub extend TensorTransferSegment with Eq::{not_equal, equal}

///|
pub extend TensorTransferRecipe with @debug.Debug::{to_repr}

///|
pub extend TensorTransferRecipe with Eq::{not_equal, equal}

///|
pub extend TensorRankPlacement with @debug.Debug::{to_repr}

///|
pub extend TensorRankPlacement with Eq::{not_equal, equal}

///|
pub extend TensorParallelOperationPlan with @debug.Debug::{to_repr}

///|
pub extend TensorParallelOperationPlan with Eq::{not_equal, equal}

///|
pub extend TensorCollectiveSequenceId with @debug.Debug::{to_repr}

///|
pub extend TensorCollectiveSequenceId with Eq::{not_equal, equal}

///|
pub extend TensorCollectiveKind with @debug.Debug::{to_repr}

///|
pub extend TensorCollectiveKind with Eq::{not_equal, equal}

///|
pub extend TensorCollectiveSite with @debug.Debug::{to_repr}

///|
pub extend TensorCollectiveSite with Eq::{not_equal, equal}

///|
pub extend TensorParallelRankPlan with @debug.Debug::{to_repr}

///|
pub extend TensorParallelRankPlan with Eq::{not_equal, equal}

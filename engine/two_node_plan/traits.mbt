///|
pub extend TwoNodePlanRule with Eq::{equal, not_equal}

///|
pub extend TwoNodePlanRule with @debug.Debug::{to_repr}

///|
pub extend TwoNodePlanError with Eq::{equal, not_equal}

///|
pub extend TwoNodePlanError with @debug.Debug::{to_repr}

///|
pub extend RankMemoryBudget with Eq::{equal, not_equal}

///|
pub extend RankMemoryBudget with @debug.Debug::{to_repr}

///|
pub extend RankCapacity with Eq::{equal, not_equal}

///|
pub extend RankCapacity with @debug.Debug::{to_repr}

///|
pub extend TwoNodePlan with Eq::{equal, not_equal}

///|
pub extend TwoNodePlan with @debug.Debug::{to_repr}

///|
pub extend TwoNodePlanDigest with Eq::{equal, not_equal}

///|
pub extend TwoNodePlanDigest with @debug.Debug::{to_repr}

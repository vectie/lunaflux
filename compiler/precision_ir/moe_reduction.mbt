///|
/// Placement and publication precision of a tensor-partitioned MoE sum.
/// These are distinct floating-point programs, not algebraic rewrite rules.
/// LocalCombinedBFloat16 publishes each rank's expert results with the existing
/// expert laws, combines routed/shared locally, then sums BF16 hidden vectors.
/// It requires both branches to be partitioned (never replicated shared work).
pub(all) enum MoeReductionPlacement {
  ExpertPartialsFloat32
  LocalCombinedBFloat16
} derive(Debug, Eq)

///|
pub extend MoeReductionPlacement with Eq::{equal, not_equal}

///|
pub extend MoeReductionPlacement with @debug.Debug::{to_repr}

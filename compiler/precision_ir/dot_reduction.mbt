///|
/// BF16 operand products are exact in F32. These laws differ only in how
/// those products are summed, retaining the same final BF16 rounding point.
/// Compensation is an explicit numerical policy, not ordered-sum equivalence.
/// Finite cancellation must not discard contributions already accumulated.
pub(all) enum DotReductionLaw {
  OrderedFloat32
  CompensatedFloat32
} derive(Debug, Eq)

///|
pub extend DotReductionLaw with @debug.Debug::{to_repr}

///|
pub extend DotReductionLaw with Eq::{equal, not_equal}

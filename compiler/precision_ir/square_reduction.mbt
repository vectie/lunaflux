///|
/// Summation accuracy is independent of storage and normalized-value rounding.
/// Both laws square in F32. Compensation retains contributions lost to a large
/// running total; it does not authorize reassociation of an ordered reduction.
pub(all) enum SquareReductionLaw {
  OrderedFloat32
  CompensatedFloat32
} derive(Debug, Eq)

///|
pub extend SquareReductionLaw with @debug.Debug::{to_repr}

///|
pub extend SquareReductionLaw with Eq::{equal, not_equal}

///|
/// Summation accuracy is independent of storage and normalized-value rounding.
/// Both laws square in F32. Compensation retains contributions lost to a large
/// running total; it does not authorize reassociation of an ordered reduction.
/// F32 overflow remains infinity; a compensation term must not turn that
/// overflow into NaN. This law does not silently widen the square/sum domain.
pub(all) enum SquareReductionLaw {
  OrderedFloat32
  CompensatedFloat32
} derive(Debug, Eq)

///|
pub extend SquareReductionLaw with @debug.Debug::{to_repr}

///|
pub extend SquareReductionLaw with Eq::{equal, not_equal}

///|
/// Summation accuracy is independent of storage and normalized-value rounding.
/// Both laws square in F32. Compensation retains contributions lost to a large
/// running total; it does not authorize reassociation of an ordered reduction.
/// F32 overflow remains infinity; a compensation term must not turn that
/// overflow into NaN. This law does not silently widen the square/sum domain.
pub(all) enum SquareReductionLaw {
  OrderedFloat32
  CompensatedFloat32
  /// Partition p owns vectors p,p+P,...; component sums join left-to-right,
  /// then partition totals join upper-half into lower-half in a fixed tree.
  /// Empty/tail components contribute zero. All squares and sums remain F32.
  VectorizedTreeFloat32(partitions~ : Int, vector_width~ : Int)
} derive(Debug, Eq)

///|
pub extend SquareReductionLaw with @debug.Debug::{to_repr}

///|
pub extend SquareReductionLaw with Eq::{equal, not_equal}

///|
pub(all) enum ReciprocalRootLaw {
  DivideRoundedSqrtFloat32
  ReciprocalSqrtFloat32
} derive(Debug, Eq)

///|
pub extend ReciprocalRootLaw with @debug.Debug::{to_repr}

///|
pub extend ReciprocalRootLaw with Eq::{equal, not_equal}

///|
fn SquareReductionLaw::validate(self : Self) -> Unit raise PrecisionPlanError {
  if self is VectorizedTreeFloat32(partitions~, vector_width~) {
    guard partitions > 0 &&
      (partitions & (partitions - 1)) == 0 &&
      vector_width > 0 &&
      (vector_width & (vector_width - 1)) == 0 &&
      partitions.to_int64() * vector_width.to_int64() <= 2147483647L else {
      raise IllegalPlacement
    }
  }
}

///|
/// BF16 operand products are exact in F32. These laws differ only in how
/// those products are summed, retaining the same final BF16 rounding point.
/// Compensation is an explicit numerical policy, not ordered-sum equivalence.
/// Finite cancellation must not discard contributions already accumulated.
pub(all) enum DotReductionLaw {
  OrderedFloat32
  CompensatedFloat32
  // Partition p owns p,p+P,...; repeatedly add the upper half into the lower
  // half before the unchanged BF16 publication. No device geometry in this law.
  StripedTreeFloat32(partitions~ : Int)
} derive(Debug, Eq)

///|
pub extend DotReductionLaw with @debug.Debug::{to_repr}

///|
pub extend DotReductionLaw with Eq::{equal, not_equal}

///|
fn DotReductionLaw::validate(self : Self, width : Int) -> Unit raise {
  match self {
    OrderedFloat32 | CompensatedFloat32 => ()
    StripedTreeFloat32(partitions~) => {
      guard partitions >= 2 &&
        partitions <= width &&
        (partitions & (partitions - 1)) == 0 &&
        width % partitions == 0 else {
        raise PrecisionPlanError::IllegalPlacement
      }
    }
  }
}

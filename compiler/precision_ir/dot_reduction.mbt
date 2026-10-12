///|
/// Products have an explicit F32 rounding boundary; BF16×BF16 products happen
/// to be exact there. These laws change summation, not product precision or
/// the enclosing operator's publication boundary.
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
fn DotReductionLaw::validate(
  self : Self,
  width : Int,
) -> Unit raise PrecisionPlanError {
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

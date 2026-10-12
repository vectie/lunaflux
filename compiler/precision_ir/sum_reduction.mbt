///|
/// Exact F32 summation order, independent of thread/subgroup implementation.
/// Butterfly partitions a power-of-two vector by descending halves before
/// joining its two subtrees. No changed exponent or division arithmetic.
pub(all) enum SumReductionLaw {
  OrderedFloat32
  ButterflyFloat32
} derive(Debug, Eq)

///|
pub extend SumReductionLaw with @debug.Debug::{to_repr}

///|
pub extend SumReductionLaw with Eq::{equal, not_equal}

///|
fn SumReductionLaw::validate(
  self : Self,
  width : Int,
) -> Unit raise PrecisionPlanError {
  guard width > 0 else { raise IllegalPlacement }
  if self is ButterflyFloat32 && (width & (width - 1)) != 0 {
    raise IllegalPlacement
  }
}

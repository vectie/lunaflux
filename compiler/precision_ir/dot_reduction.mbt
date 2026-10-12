///|
/// Products and accumulation have explicit F32 rounding boundaries.
/// Ordered/compensated/striped laws round products separately; grouped-vector
/// laws explicitly choose separate rounding or one FMA rounding per update.
/// None change the enclosing operator's publication boundary.
/// Compensation is an explicit numerical policy, not ordered-sum equivalence.
/// Finite cancellation must not discard contributions already accumulated.
pub(all) enum DotReductionLaw {
  OrderedFloat32
  CompensatedFloat32
  // Partition p owns p,p+P,...; repeatedly add the upper half into the lower
  // half before the unchanged BF16 publication. No device geometry in this law.
  StripedTreeFloat32(partitions~ : Int)
  // Group g, partition p, component v owns ((g*P+p)*V+v)+k*G*P*V.
  // Component sums join left-to-right, then each group's P partitions reduce
  // by descending halves. Publication partition q adds g=q,q+Q,... in order;
  // the Q totals join left-to-right. Empty positions are zero identities.
  GroupedVectorFloat32(
    groups~ : Int,
    partitions~ : Int,
    vector_width~ : Int,
    publication_partitions~ : Int,
    multiply_add~ : Float32AffineLaw
  )
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
    GroupedVectorFloat32(
      groups~,
      partitions~,
      vector_width~,
      publication_partitions~,
      ..
    ) => {
      guard width > 0 &&
        groups > 0 &&
        partitions > 0 &&
        (partitions & (partitions - 1)) == 0 &&
        vector_width > 0 &&
        (vector_width & (vector_width - 1)) == 0 &&
        publication_partitions > 0 &&
        publication_partitions <= groups &&
        groups.to_int64() * partitions.to_int64() <=
        2147483647L / vector_width.to_int64() else {
        raise PrecisionPlanError::IllegalPlacement
      }
    }
  }
}

///|
/// A partitioned softmax state is `(maximum, denominator, numerator[])`.
/// Statistics are invariant across output components: one owner computes
/// them, publishes them once, then disjoint component owners apply the same
/// ascending-partition numerator fold. Sharing does not reassociate sums.
pub struct PartitionMerge {
  priv partitions : Int
  priv dimensions : Int
} derive(Debug, Eq)

///|
pub extend PartitionMerge with Eq::{equal, not_equal}

///|
pub extend PartitionMerge with @debug.Debug::{to_repr}

///|
pub fn PartitionMerge::new(
  partitions~ : Int,
  dimensions~ : Int,
) -> PartitionMerge? {
  if partitions > 1 && dimensions > 0 {
    Some({ partitions, dimensions, })
  } else {
    None
  }
}

///|
pub fn PartitionMerge::dimensions(self : PartitionMerge) -> (Int, Int) {
  (self.partitions, self.dimensions)
}

///|
pub(all) enum PartitionMergeStep {
  ReducePartitionMaximum
  FormPartitionScalesAndDenominator
  PublishPartitionStatistics
  AccumulatePartitionNumerators
  NormalizePartitionNumerators
} derive(Debug, Eq)

///|
pub extend PartitionMergeStep with Eq::{equal, not_equal}

///|
pub extend PartitionMergeStep with @debug.Debug::{to_repr}

///|
pub fn PartitionMerge::steps(
  _self : PartitionMerge,
) -> ReadOnlyArray[PartitionMergeStep] {
  [
    ReducePartitionMaximum,
    FormPartitionScalesAndDenominator,
    PublishPartitionStatistics,
    AccumulatePartitionNumerators,
    NormalizePartitionNumerators,
  ]
}

///|
pub fn PartitionMerge::numeric_law(_self : PartitionMerge) -> String {
  "ascending-partition-f32-statistics-v1"
}

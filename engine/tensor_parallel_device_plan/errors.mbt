///|
pub(all) enum TensorParallelDevicePlanRule {
  DeviceOrdinal
  Rank
  WorldSize
  ModelIdentity
  MaterializedRankPlan
  DeviceTarget
  OperationLayout
  TensorCount
  TensorReference
  TensorPlacement
  TensorExtent
  TensorRegion
  FullShardedTensor
  CollectiveCount
  CollectiveSequence
  CollectiveOperation
  CollectiveKind
  CollectiveWidth
} derive(Debug, Eq)

///|
pub(all) suberror TensorParallelDevicePlanError {
  InvalidTensorParallelDevicePlan(
    rule~ : TensorParallelDevicePlanRule,
    index~ : Int
  )
} derive(Debug, Eq)

///|
fn invalid(
  rule : TensorParallelDevicePlanRule,
  index : Int,
) -> TensorParallelDevicePlanError {
  InvalidTensorParallelDevicePlan(rule~, index~)
}

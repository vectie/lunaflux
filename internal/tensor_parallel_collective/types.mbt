///|
pub(all) enum TensorParallelCollectiveStage {
  RankPlanIdentity
  RankPlanShape
  CanonicalContract
  DeviceContract
} derive(Debug, Eq)

///|
pub(all) suberror TensorParallelCollectiveError {
  InvalidRankPlan(TensorParallelCollectiveStage)
  Device(TensorParallelCollectiveStage, @device.DeviceError)
} derive(Debug, Eq)

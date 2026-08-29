///|
priv struct PreparedShardedStreamingLoad[S] {
  source_stamp : S
  payload_offset_bytes : Int64
  rank_plan : @tensor_parallel_plan.TensorParallelRankPlan
  source_order : FixedArray[Int]
  scratch : FixedArray[Byte]
}

///|
pub struct DeviceShardedWeightFileInspection {
  priv model : @spec.LlamaModelMetadata
  priv semantic_plan : @plan.ModelPlan
  priv locator : @approved_fs.ApprovedRelativeLocator
  priv limits : DeviceWeightFileLimits
  priv max_arena_bytes : Int64
  priv payload_offset_bytes : Int64
  priv rank_plan : @tensor_parallel_plan.TensorParallelRankPlan
}

///|
pub struct DeviceShardedWeights {
  priv allocation : @device.Allocation
  priv rank_plan : @tensor_parallel_plan.TensorParallelRankPlan
}

///|
pub(all) enum DeviceShardedWeightPreparation {
  ShardedWeightsReady(DeviceShardedWeights)
  ShardedWeightsCleanupRequired(FailedDeviceShardedWeightPreparation)
}

///|
pub struct FailedDeviceShardedWeightPreparation {
  priv allocation : @device.Allocation
  priv rank_plan : @tensor_parallel_plan.TensorParallelRankPlan
  priv cause : DeviceWeightError
  priv stage : DeviceWeightFailureStage
  priv mut cleanup_failure : @device.DeviceError
  priv mut open : Bool
}

///|
priv struct LoadedShardedResource[T] {
  resource : T
  rank_plan : @tensor_parallel_plan.TensorParallelRankPlan
}

///|
priv enum ShardedResourcePreparation[T] {
  ShardedResourceReady(LoadedShardedResource[T])
  ShardedResourceCleanupRequired(FailedShardedResourcePreparation[T])
}

///|
priv struct FailedShardedResourcePreparation[T] {
  resource : T
  rank_plan : @tensor_parallel_plan.TensorParallelRankPlan
  cause : DeviceWeightError
  stage : DeviceWeightFailureStage
  cleanup_failure : @device.DeviceError
}

///|
priv struct AbsoluteShardedTransfer {
  source_offset_bytes : Int64
  destination_offset_bytes : Int64
  byte_length : Int64
}

///|
priv struct ShardedTransferCursor {
  mut order_index : Int
  mut segment : Int64
  mut reference : Int
  mut source_offset_bytes : Int64
  mut destination_offset_bytes : Int64
  mut byte_length : Int64
}

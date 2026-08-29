///|
pub fn DeviceShardedWeightFileInspection::content_digest(
  self : DeviceShardedWeightFileInspection,
) -> @spec.ContentDigest {
  self.rank_plan.identity().content()
}

///|
pub fn DeviceShardedWeightFileInspection::rank_plan(
  self : DeviceShardedWeightFileInspection,
) -> @tensor_parallel_plan.TensorParallelRankPlan {
  self.rank_plan
}

///|
pub fn DeviceShardedWeights::identity(
  self : DeviceShardedWeights,
) -> @spec.ModelIdentity {
  self.rank_plan.identity()
}

///|
pub fn DeviceShardedWeights::rank_plan(
  self : DeviceShardedWeights,
) -> @tensor_parallel_plan.TensorParallelRankPlan {
  self.rank_plan
}

///|
pub fn DeviceShardedWeights::allocation(
  self : DeviceShardedWeights,
) -> @device.Allocation {
  self.allocation
}

///|
pub fn DeviceShardedWeights::close(
  self : DeviceShardedWeights,
) -> Unit raise DeviceWeightError {
  self.allocation.close() catch {
    error => raise DeviceFailed(error)
  }
}

///|
pub fn FailedDeviceShardedWeightPreparation::cause(
  self : FailedDeviceShardedWeightPreparation,
) -> DeviceWeightError {
  self.cause
}

///|
pub fn FailedDeviceShardedWeightPreparation::stage(
  self : FailedDeviceShardedWeightPreparation,
) -> DeviceWeightFailureStage {
  self.stage
}

///|
pub fn FailedDeviceShardedWeightPreparation::rank_plan(
  self : FailedDeviceShardedWeightPreparation,
) -> @tensor_parallel_plan.TensorParallelRankPlan {
  self.rank_plan
}

///|
pub fn FailedDeviceShardedWeightPreparation::cleanup_failure(
  self : FailedDeviceShardedWeightPreparation,
) -> @device.DeviceError {
  self.cleanup_failure
}

///|
pub fn FailedDeviceShardedWeightPreparation::is_closed(
  self : FailedDeviceShardedWeightPreparation,
) -> Bool {
  !self.open
}

///|
pub fn FailedDeviceShardedWeightPreparation::retry_close(
  self : FailedDeviceShardedWeightPreparation,
) -> Unit raise DeviceWeightError {
  if !self.open {
    return
  }
  self.allocation.close() catch {
    error => {
      self.cleanup_failure = error
      raise CleanupFailed(stage=self.stage, error)
    }
  }
  self.open = false
}

///|
pub fn TensorParallelDeviceWorkerOwner::lifecycle(
  self : TensorParallelDeviceWorkerOwner,
) -> TensorParallelDeviceWorkerLifecycle {
  self.lifecycle
}

///|
pub fn TensorParallelDeviceWorkerOwner::failure_class(
  self : TensorParallelDeviceWorkerOwner,
) -> TensorParallelDeviceWorkerFailureClass? {
  self.failure_class
}

///|
pub fn TensorParallelDeviceWorkerOwner::primary_failure(
  self : TensorParallelDeviceWorkerOwner,
) -> TensorParallelDeviceWorkerError? {
  self.primary_failure
}

///|
pub fn TensorParallelDeviceWorkerOwner::cleanup_failure(
  self : TensorParallelDeviceWorkerOwner,
) -> TensorParallelDeviceWorkerError? {
  self.cleanup_failure
}

///|
pub fn TensorParallelDeviceWorkerOwner::readiness_contract(
  self : TensorParallelDeviceWorkerOwner,
) -> @worker_wire.WorkerStartupContract raise TensorParallelDeviceWorkerError {
  if self.lifecycle != WorkerReady || !resources_complete(self.resources) {
    raise Invalid(ResourceState)
  }
  self.plan.startup
}

///|
pub fn FailedTensorParallelDeviceWorkerPreparation::cause(
  self : FailedTensorParallelDeviceWorkerPreparation,
) -> TensorParallelDeviceWorkerError {
  self.cause
}

///|
pub fn FailedTensorParallelDeviceWorkerPreparation::cleanup_failure(
  self : FailedTensorParallelDeviceWorkerPreparation,
) -> TensorParallelDeviceWorkerError {
  self.cleanup_failure
}

///|
pub fn FailedTensorParallelDeviceWorkerPreparation::is_closed(
  self : FailedTensorParallelDeviceWorkerPreparation,
) -> Bool {
  self.lifecycle == WorkerClosed
}

///|
pub fn TensorParallelFollowerExecutionAck::group_generation_value(
  self : TensorParallelFollowerExecutionAck,
) -> UInt64 {
  self.group_generation_value
}

///|
pub fn TensorParallelFollowerExecutionAck::model_generation_value(
  self : TensorParallelFollowerExecutionAck,
) -> UInt64 {
  self.model_generation_value
}

///|
pub fn TensorParallelFollowerExecutionAck::plan_sequence_value(
  self : TensorParallelFollowerExecutionAck,
) -> UInt64 {
  self.plan_sequence_value
}

///|
pub fn TensorParallelFollowerExecutionAck::rank(
  self : TensorParallelFollowerExecutionAck,
) -> Int {
  self.rank
}

///|
pub fn TensorParallelFollowerExecutionAck::world_size(
  self : TensorParallelFollowerExecutionAck,
) -> Int {
  self.world_size
}

///|
pub fn TensorParallelDeviceExecutionProgress::is_pending(
  self : TensorParallelDeviceExecutionProgress,
) -> Bool {
  !self.4
}

///|
pub fn TensorParallelDeviceExecutionProgress::executed(
  self : TensorParallelDeviceExecutionProgress,
) -> ExecutedTensorParallelDeviceExecution raise TensorParallelDeviceWorkerError {
  if !self.4 ||
    self.1 == 0UL ||
    self.0.lifecycle != WorkerExecuted ||
    self.0.active_epoch != self.1 {
    raise Invalid(PlanFrame)
  }
  ExecutedTensorParallelDeviceExecution(self.0, self.1, self.2, self.3)
}

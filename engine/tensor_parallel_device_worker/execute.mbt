///|
fn latch_worker_failure(
  owner : TensorParallelDeviceWorkerOwner,
  class : TensorParallelDeviceWorkerFailureClass,
  error : TensorParallelDeviceWorkerError,
) -> Unit {
  if owner.primary_failure is None {
    owner.primary_failure = Some(error)
    owner.failure_class = Some(class)
  }
  owner.resources.collective.abort_required = true
  owner.resources.executor.abort_required = true
  owner.lifecycle = WorkerFaulted
}

///|
fn fail_rank_execution(
  owner : TensorParallelDeviceWorkerOwner,
  error : TensorParallelDeviceWorkerError,
) -> TensorParallelDeviceWorkerError {
  latch_worker_failure(owner, RankExecutionFailure, error)
  error
}

///|
fn fail_collective(
  owner : TensorParallelDeviceWorkerOwner,
  error : TensorParallelDeviceWorkerError,
) -> TensorParallelDeviceWorkerError {
  latch_worker_failure(owner, CollectiveFailure, error)
  error
}

///|
fn authenticate_staged_execution(
  staged : StagedTensorParallelDeviceExecution,
  owner : TensorParallelDeviceWorkerOwner,
  frame : @worker_wire.ValidatedPlanFrame,
) -> @device_step.StagedDeviceStep raise TensorParallelDeviceWorkerError {
  if !physical_equal(staged.0, owner) ||
    staged.1 != owner.active_epoch ||
    (
      owner.lifecycle != WorkerStaged &&
      owner.lifecycle != WorkerExecuting &&
      owner.lifecycle != WorkerCollectivePending &&
      owner.lifecycle != WorkerDevicePending
    ) {
    raise Invalid(PlanFrame)
  }
  let matches = frame.matches_owner_epoch(staged.2, staged.3) catch {
    error => raise Wire(error)
  }
  if !matches {
    raise Invalid(PlanFrame)
  }
  let exact = owner.resources.descriptor.rederive_staged(staged.1) catch {
    error => raise Descriptor(error)
  }
  exact
}

///|
fn authenticate_executed_execution(
  executed : ExecutedTensorParallelDeviceExecution,
  owner : TensorParallelDeviceWorkerOwner,
  frame : @worker_wire.ValidatedPlanFrame,
) -> @device_step.StagedDeviceStep raise TensorParallelDeviceWorkerError {
  if !physical_equal(executed.0, owner) ||
    executed.1 != owner.active_epoch ||
    owner.lifecycle != WorkerExecuted {
    raise Invalid(PlanFrame)
  }
  let matches = frame.matches_owner_epoch(executed.2, executed.3) catch {
    error => raise Wire(error)
  }
  if !matches {
    raise Invalid(PlanFrame)
  }
  let exact = owner.resources.descriptor.rederive_staged(executed.1) catch {
    error => raise Descriptor(error)
  }
  exact
}

///|
pub fn TensorParallelDeviceWorkerOwner::poll_startup(
  self : TensorParallelDeviceWorkerOwner,
) -> Bool raise TensorParallelDeviceWorkerError {
  match self.lifecycle {
    WorkerStarting => ()
    WorkerReady => return true
    _ => raise Invalid(ResourceState)
  }
  let ready = self.resources.collective.resource.poll_ready() catch {
    error => {
      let translated = Device(Collective, error)
      raise fail_collective(self, translated)
    }
  }
  if ready {
    self.resources.collective.abort_required = false
    self.lifecycle = WorkerReady
  }
  ready
}

///|
/// Authenticates and uploads one exact plan frame into the reusable rank-local
/// descriptor owner. Only scalar epoch/count/cursor state is retained.
pub fn TensorParallelDeviceWorkerOwner::stage_frame(
  self : TensorParallelDeviceWorkerOwner,
  frame : @worker_wire.ValidatedPlanFrame,
) -> StagedTensorParallelDeviceExecution raise TensorParallelDeviceWorkerError {
  if self.lifecycle != WorkerReady {
    raise Invalid(ResourceState)
  }
  let frame_owner = frame.owner_identity() catch {
    error => {
      let translated = Wire(error)
      raise fail_rank_execution(self, translated)
    }
  }
  let frame_epoch = frame.epoch_value() catch {
    error => {
      let translated = Wire(error)
      raise fail_rank_execution(self, translated)
    }
  }
  let staged = self.resources.descriptor.stage_frame(frame) catch {
    error => {
      let translated = TensorParallelDeviceWorkerError::Descriptor(error)
      raise fail_rank_execution(self, translated)
    }
  }
  staged.detach_frame_authentication(frame) catch {
    error => {
      let translated = TensorParallelDeviceWorkerError::Descriptor(error)
      raise fail_rank_execution(self, translated)
    }
  }
  let sequence = staged.sequence_value() catch {
    error => {
      let translated = TensorParallelDeviceWorkerError::Descriptor(error)
      raise fail_rank_execution(self, translated)
    }
  }
  let rows = staged.row_count() catch {
    error => {
      let translated = TensorParallelDeviceWorkerError::Descriptor(error)
      raise fail_rank_execution(self, translated)
    }
  }
  let tokens = staged.token_count() catch {
    error => {
      let translated = TensorParallelDeviceWorkerError::Descriptor(error)
      raise fail_rank_execution(self, translated)
    }
  }
  if sequence == 0UL || rows <= 0 || tokens <= 0 {
    let error = Invalid(PlanFrame)
    raise fail_rank_execution(self, error)
  }
  self.active_epoch = staged.epoch_value()
  self.active_sequence = sequence
  self.active_rows = rows
  self.active_tokens = tokens
  self.operation_index = 0
  self.collective_index = 0
  self.resources.collective.abort_required = true
  self.resources.executor.abort_required = false
  self.lifecycle = WorkerStaged
  StagedTensorParallelDeviceExecution(
    self,
    self.active_epoch,
    frame_owner,
    frame_epoch,
  )
}

///|
fn collective_kind(
  kind : @tensor_parallel_execution_plan.TensorParallelPhysicalCollectiveKind,
) -> @device.TensorCollectiveKind {
  match kind {
    InPlaceSumAllReduce => Bf16SumAllReduce
    InPlaceLastAxisAllGather => Bf16AllGather
  }
}

///|
fn submit_current_collective(
  owner : TensorParallelDeviceWorkerOwner,
  collective : @tensor_parallel_execution_plan.TensorParallelCollectiveBinding,
) -> Unit raise TensorParallelDeviceWorkerError {
  if owner.next_global_collective_sequence == 18446744073709551615UL {
    let error = Invalid(SequenceOverflow)
    raise fail_collective(owner, error)
  }
  let activation = owner.resources.activation_workspace.resource
  owner.resources.collective.resource.submit(
    generation=owner.plan.group_generation.value(),
    plan_sequence=owner.active_sequence,
    global_sequence=owner.next_global_collective_sequence,
    site_sequence=collective.sequence(),
    operation_id=collective.operation_id().as_int(),
    kind=collective_kind(collective.kind()),
    live_query_tokens=owner.active_tokens,
    send=activation,
    send_offset=collective.local_send_offset_bytes(),
    receive=activation,
    receive_offset=collective.region().offset_bytes(),
    stream=owner.resources.stream.resource,
  ) catch {
    error => {
      let translated = Device(Collective, error)
      raise fail_collective(owner, translated)
    }
  }
  owner.lifecycle = WorkerCollectivePending
}

///|
fn poll_current_collective(
  owner : TensorParallelDeviceWorkerOwner,
) -> Bool raise TensorParallelDeviceWorkerError {
  let complete = owner.resources.collective.resource.poll() catch {
    error => {
      let translated = Device(Collective, error)
      raise fail_collective(owner, translated)
    }
  }
  if complete {
    owner.collective_index += 1
    owner.next_global_collective_sequence += 1UL
    owner.lifecycle = WorkerExecuting
  }
  complete
}

///|
fn poll_device_completion(
  owner : TensorParallelDeviceWorkerOwner,
) -> Bool raise TensorParallelDeviceWorkerError {
  let complete = owner.resources.executor.resource.poll_completion() catch {
    error => {
      let translated = Device(KernelExecutor, error)
      raise fail_rank_execution(owner, translated)
    }
  }
  if complete {
    owner.resources.executor.abort_required = false
    owner.lifecycle = WorkerExecuted
  }
  complete
}

///|
/// Advances one rank through its prebuilt ordered AOT launches and exact
/// collective transcript. Native collectives are polled; false progress never
/// blocks or allocates and never commits a sequence.
pub fn TensorParallelDeviceWorkerOwner::progress_execution(
  self : TensorParallelDeviceWorkerOwner,
  staged : StagedTensorParallelDeviceExecution,
  frame : @worker_wire.ValidatedPlanFrame,
) -> TensorParallelDeviceExecutionProgress raise TensorParallelDeviceWorkerError {
  ignore(authenticate_staged_execution(staged, self, frame))
  if self.lifecycle == WorkerDevicePending {
    return TensorParallelDeviceExecutionProgress(
      self,
      self.active_epoch,
      staged.2,
      staged.3,
      poll_device_completion(self),
    )
  }
  if self.lifecycle == WorkerCollectivePending && !poll_current_collective(self) {
    return TensorParallelDeviceExecutionProgress(
      self,
      self.active_epoch,
      staged.2,
      staged.3,
      false,
    )
  }
  self.lifecycle = WorkerExecuting
  while self.operation_index < self.steps.length() {
    let step = self.steps[self.operation_index]
    self.resources.executor.resource.enqueue(self.operation_index) catch {
      error => {
        let translated = Device(KernelExecutor, error)
        raise fail_rank_execution(self, translated)
      }
    }
    self.resources.executor.abort_required = true
    self.operation_index += 1
    if self.collective_index < self.plan.execution.collective_count() {
      let collective = self.plan.execution.collective_exact(
        self.collective_index,
      ) catch {
        _ => {
          let error = Invalid(CollectiveOrder)
          raise fail_rank_execution(self, error)
        }
      }
      if collective.operation_id() == step.operation_id {
        submit_current_collective(self, collective)
        return TensorParallelDeviceExecutionProgress(
          self,
          self.active_epoch,
          staged.2,
          staged.3,
          false,
        )
      }
    }
  }
  if self.collective_index != self.plan.execution.collective_count() {
    let error = Invalid(CollectiveOrder)
    raise fail_rank_execution(self, error)
  }
  self.resources.executor.resource.record_completion() catch {
    error => {
      let translated = Device(KernelExecutor, error)
      raise fail_rank_execution(self, translated)
    }
  }
  self.lifecycle = WorkerDevicePending
  TensorParallelDeviceExecutionProgress(
    self,
    self.active_epoch,
    staged.2,
    staged.3,
    false,
  )
}

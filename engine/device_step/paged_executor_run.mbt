///|
fn require_paged_lifecycle(
  actual : PagedGraphLifecycle,
  expected : PagedGraphLifecycle,
) -> Unit raise DeviceStepError {
  if actual != expected {
    raise InvalidExecutorLifecycle(expected~, actual~)
  }
}

///|
fn complete_descriptor(
  resources : PagedGraphResources,
) -> DeviceStepOwner raise DeviceStepError {
  guard resources.descriptor is Some(Complete(owner)) else {
    raise InvalidExecutor(ResourceState)
  }
  owner
}

///|
fn open_paged_stream(
  resources : PagedGraphResources,
) -> @device.Stream raise DeviceStepError {
  guard resources.stream is Some(owner) && owner.open else {
    raise InvalidExecutor(ResourceState)
  }
  owner.resource
}

///|
fn authenticate_staged_paged(
  staged : StagedPagedGraphExecution,
  executor : PagedGraphExecutor,
) -> DeviceStepOwner raise DeviceStepError {
  if !physical_equal(staged.executor, executor) {
    raise ExecutorOwnerMismatch
  }
  require_paged_lifecycle(executor.lifecycle, GraphStaged)
  let descriptor = complete_descriptor(executor.resources)
  if staged.epoch != descriptor.stage_epoch {
    raise ExecutorEpochMismatch
  }
  StagedDeviceStep(descriptor, staged.epoch).authenticate(descriptor)
  descriptor
}

///|
fn authenticate_executed_paged(
  executed : ExecutedPagedGraphExecution,
  executor : PagedGraphExecutor,
) -> DeviceStepOwner raise DeviceStepError {
  if !physical_equal(executed.executor, executor) {
    raise ExecutorOwnerMismatch
  }
  require_paged_lifecycle(executor.lifecycle, GraphExecuted)
  let descriptor = complete_descriptor(executor.resources)
  if executed.epoch != descriptor.stage_epoch {
    raise ExecutorEpochMismatch
  }
  StagedDeviceStep(descriptor, executed.epoch).authenticate(descriptor)
  descriptor
}

///|
/// Validates and publishes one authenticated schedule into the executor's
/// private reusable descriptor storage.
pub fn PagedGraphExecutor::stage(
  self : PagedGraphExecutor,
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> StagedPagedGraphExecution raise DeviceStepError {
  require_paged_lifecycle(self.lifecycle, GraphReady)
  let descriptor = complete_descriptor(self.resources)
  let staged = descriptor.stage(plan) catch {
    error => {
      if descriptor.lifecycle() == Poisoned {
        self.lifecycle = GraphPoisoned
      }
      raise error
    }
  }
  self.lifecycle = GraphStaged
  { executor: self, epoch: staged.1 }
}

///|
/// Launches every admitted full-graph AOT step in semantic order. Each native
/// launch synchronizes before returning. Any failure permanently poisons this
/// owner because preceding operations may already have mutated activation or
/// persistent KV storage.
pub fn PagedGraphExecutor::execute(
  self : PagedGraphExecutor,
  staged : StagedPagedGraphExecution,
) -> ExecutedPagedGraphExecution raise DeviceStepError {
  let descriptor = authenticate_staged_paged(staged, self)
  let stream = open_paged_stream(self.resources)
  self.lifecycle = GraphExecuting
  let mut index = 0
  while index < self.steps.length() {
    let step = self.steps[index]
    guard self.resources.functions.get(step.function_index) is Some(function) &&
      function.open else {
      self.lifecycle = GraphPoisoned
      descriptor.lifecycle = Poisoned
      raise InvalidExecutor(ResourceState)
    }
    function.resource.launch_synchronous(
      step.dimensions,
      step.arguments,
      stream,
    ) catch {
      error => {
        self.lifecycle = GraphPoisoned
        descriptor.lifecycle = Poisoned
        raise ExecutorLaunchFailed(step.operation_id, error)
      }
    }
    index += 1
  }
  self.lifecycle = GraphExecuted
  { executor: self, epoch: staged.epoch }
}

///|
/// Retires the exact successful execution and advances the accepted plan
/// predecessor. A staged but unexecuted or poisoned graph cannot finish.
pub fn PagedGraphExecutor::finish(
  self : PagedGraphExecutor,
  executed : ExecutedPagedGraphExecution,
) -> Unit raise DeviceStepError {
  let descriptor = authenticate_executed_paged(executed, self)
  descriptor.finish(StagedDeviceStep(descriptor, executed.epoch))
  self.lifecycle = GraphReady
}

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
fn open_paged_ordered_executor(
  resources : PagedGraphResources,
  index : Int,
) -> @device.OrderedKernelExecutor raise DeviceStepError {
  guard resources.ordered_executors.get(index) is Some(owner) && owner.open else {
    raise InvalidExecutor(ResourceState)
  }
  owner.resource
}

///|
fn abort_paged_ordered_executor(
  executor : @device.OrderedKernelExecutor,
) -> Unit {
  try executor.abort() catch {
    _ => ()
  } noraise {
    _ => ()
  }
}

///|
fn poison_fp8_graph_lifecycle(executor : PagedGraphExecutor) -> Unit {
  clear_fp8_frame_admission_v3(executor)
  executor.lifecycle = GraphPoisoned
}

///|
fn poison_fp8_ordered_execution(
  executor : PagedGraphExecutor,
  descriptor : DeviceStepOwner,
  ordered : @device.OrderedKernelExecutor,
  error : DeviceStepError,
) -> Unit raise DeviceStepError {
  poison_fp8_graph_lifecycle(executor)
  descriptor.lifecycle = Poisoned
  abort_paged_ordered_executor(ordered)
  raise error
}

///|
fn[T] propagate_descriptor_poison(
  executor : PagedGraphExecutor,
  descriptor : DeviceStepOwner,
  error : DeviceStepError,
) -> T raise DeviceStepError {
  if descriptor.lifecycle() == Poisoned {
    executor.lifecycle = GraphPoisoned
  }
  raise error
}

///|
fn poison_fp8_frame_publication(
  executor : PagedGraphExecutor,
  descriptor : DeviceStepOwner,
  error : DeviceStepError,
) -> Unit raise DeviceStepError {
  poison_fp8_graph_lifecycle(executor)
  descriptor.lifecycle = Poisoned
  raise error
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
  if self.fp8_frame_envelope is Some(_) {
    raise InvalidExecutor(Fp8FrameAdmission)
  }
  let graph_bucket = select_submitted_graph_bucket(
    self.graph_bucket_table,
    plan,
  )
  let ordered_executor_index = self.ordered_dispatch.select(graph_bucket)
  let descriptor = complete_descriptor(self.resources)
  let staged = descriptor.stage_with_sampling_mode(
    plan,
    self.completion.mode is CudaGreedyCompletion(_),
  ) catch {
    error => propagate_descriptor_poison(self, descriptor, error)
  }
  stage_paged_device_sampling_plan(self, descriptor, plan) catch {
    error => {
      poison_paged_completion(self, descriptor)
      raise error
    }
  }
  self.lifecycle = GraphStaged
  { executor: self, epoch: staged.1, graph_bucket, ordered_executor_index }
}

///|
/// Publishes one exact validated wire plan into the full graph executor. No
/// scheduler-owned mutable protocol object crosses this worker boundary.
pub fn PagedGraphExecutor::stage_frame(
  self : PagedGraphExecutor,
  frame : @worker_wire.ValidatedPlanFrame,
) -> StagedPagedGraphExecution raise DeviceStepError {
  require_paged_lifecycle(self.lifecycle, GraphReady)
  let graph_bucket = select_frame_graph_bucket(self.graph_bucket_table, frame)
  let ordered_executor_index = self.ordered_dispatch.select(graph_bucket)
  let descriptor = complete_descriptor(self.resources)
  let staged = match self.fp8_frame_envelope {
    Some(envelope) => {
      let summary = preflight_fp8_frame_v3(descriptor, frame, envelope)
      let staged = descriptor.stage_preflighted_frame(frame, summary) catch {
        error => propagate_descriptor_poison(self, descriptor, error)
      }
      authenticate_and_publish_fp8_frame_token_v3(
        self, descriptor, frame, staged, summary,
      ) catch {
        error => poison_fp8_frame_publication(self, descriptor, error)
      }
      staged
    }
    None => {
      let staged = descriptor.stage_frame_with_sampling_mode(
        frame,
        self.completion.mode is CudaGreedyCompletion(_),
      ) catch {
        error => propagate_descriptor_poison(self, descriptor, error)
      }
      stage_paged_device_sampling_frame(self, descriptor, frame) catch {
        error => {
          poison_paged_completion(self, descriptor)
          raise error
        }
      }
      clear_fp8_frame_admission_v3(self)
      staged
    }
  }
  self.lifecycle = GraphStaged
  { executor: self, epoch: staged.1, graph_bucket, ordered_executor_index }
}

///|
/// Enqueues every admitted full-graph AOT step in semantic order and waits on
/// one reusable completion event after the final launch. Any failure drains
/// the retained stream and permanently poisons this owner because preceding
/// operations may already have mutated activation or persistent KV storage.
pub fn PagedGraphExecutor::execute(
  self : PagedGraphExecutor,
  staged : StagedPagedGraphExecution,
) -> ExecutedPagedGraphExecution raise DeviceStepError {
  let descriptor = authenticate_staged_paged(staged, self)
  authenticate_and_consume_fp8_frame_token_v3(self, descriptor, staged)
  let ordered = open_paged_ordered_executor(
    self.resources,
    staged.ordered_executor_index,
  )
  arm_fused_dispatch_canary(self, descriptor) catch {
    error => poison_fp8_ordered_execution(self, descriptor, ordered, error)
  }
  arm_fp8_scale_evidence_v3(self) catch {
    error => poison_fp8_ordered_execution(self, descriptor, ordered, error)
  }
  self.lifecycle = GraphExecuting
  match ordered.execution_mode() {
    OrderedKernelCaptured =>
      ordered.launch_captured() catch {
        error => {
          self.lifecycle = GraphPoisoned
          descriptor.lifecycle = Poisoned
          abort_paged_ordered_executor(ordered)
          raise ExecutorDevice(OrderedExecutor, error)
        }
      }
    OrderedKernelEager =>
      for index in 0..<self.steps.length() {
        let step = self.steps[index]
        ordered.enqueue(index) catch {
          error => {
            self.lifecycle = GraphPoisoned
            descriptor.lifecycle = Poisoned
            abort_paged_ordered_executor(ordered)
            raise ExecutorLaunchFailed(step.operation_id, error)
          }
        }
      }
  }
  ordered.record_completion() catch {
    error => {
      self.lifecycle = GraphPoisoned
      descriptor.lifecycle = Poisoned
      abort_paged_ordered_executor(ordered)
      raise ExecutorDevice(OrderedExecutor, error)
    }
  }
  ordered.wait_completion() catch {
    error => {
      self.lifecycle = GraphPoisoned
      descriptor.lifecycle = Poisoned
      abort_paged_ordered_executor(ordered)
      raise ExecutorDevice(OrderedExecutor, error)
    }
  }
  validate_fused_dispatch_canary(self, descriptor, staged.epoch) catch {
    error => poison_fp8_ordered_execution(self, descriptor, ordered, error)
  }
  validate_fp8_scale_evidence_v3(self) catch {
    error => poison_fp8_ordered_execution(self, descriptor, ordered, error)
  }
  self.graph_telemetry.record_completed(ordered.execution_mode())
  self.lifecycle = GraphExecuted
  {
    executor: self,
    epoch: staged.epoch,
    ordered_executor_index: staged.ordered_executor_index,
  }
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
  let ordered = open_paged_ordered_executor(
    self.resources,
    executed.ordered_executor_index,
  )
  ordered.reset() catch {
    error => {
      self.lifecycle = GraphPoisoned
      descriptor.lifecycle = Poisoned
      raise ExecutorDevice(OrderedExecutor, error)
    }
  }
  self.lifecycle = GraphReady
}

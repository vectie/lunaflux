///|
fn ready_executor(
  resources : I8DeviceWorkerResources,
) -> @device_step.PagedGraphExecutor raise I8DeviceWorkerError {
  if !resources_complete(resources) {
    raise Invalid(ResourceState)
  }
  guard resources.executor is Some(Complete(executor)) else {
    raise Invalid(ResourceState)
  }
  executor
}

///|
fn[T] fail_execution(
  owner : I8DeviceWorkerOwner,
  writer : @worker_wire.CompletionFrameWriter,
  primary : I8DeviceWorkerError,
) -> T raise I8DeviceWorkerError {
  owner.lifecycle = WorkerFaulted
  try writer.abort() catch {
    abort_error => raise CompletionAbortFailed(primary~, abort=abort_error)
  } noraise {
    _ => raise primary
  }
}

///|
/// Executes exactly one authenticated frame. Any authentication, staging,
/// launch, sampling, finish, or submit failure permanently fault-stops the
/// owner while preserving deterministic cleanup authority.
pub fn I8DeviceWorkerOwner::execute_frame(
  self : I8DeviceWorkerOwner,
  frame : @worker_wire.ValidatedPlanFrame,
  writer : @worker_wire.CompletionFrameWriter,
) -> @worker_wire.ValidatedCompletionFrame raise I8DeviceWorkerError {
  if self.lifecycle != WorkerReady {
    raise InvalidLifecycle(self.lifecycle)
  }
  let executor = ready_executor(self.resources) catch {
    error => return fail_execution(self, writer, error)
  }
  writer.validate_plan(frame) catch {
    error => return fail_execution(self, writer, Wire(error))
  }
  let staged = executor.stage_frame(frame) catch {
    error => return fail_execution(self, writer, Executor(error))
  }
  let executed = executor.execute(staged) catch {
    error => return fail_execution(self, writer, Executor(error))
  }
  executor.sample_completion_frame(executed, frame, writer) catch {
    error => return fail_execution(self, writer, Executor(error))
  }
  executor.finish(executed) catch {
    error => return fail_execution(self, writer, Executor(error))
  }
  writer.submit() catch {
    error => return fail_execution(self, writer, Wire(error))
  }
}

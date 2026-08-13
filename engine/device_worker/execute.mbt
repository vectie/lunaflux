///|
fn ready_executor(
  resources : DeviceWorkerResources,
) -> @device_step.PagedGraphExecutor raise DeviceWorkerError {
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
  owner : DeviceWorkerOwner,
  writer : @worker_wire.CompletionFrameWriter,
  primary : DeviceWorkerError,
) -> T raise DeviceWorkerError {
  owner.lifecycle = WorkerFaulted
  try writer.abort() catch {
    abort_error => raise CompletionAbortFailed(primary~, abort=abort_error)
  } noraise {
    _ => raise primary
  }
}

///|
/// Executes one authenticated worker plan through the privately owned paged
/// graph and returns its canonical completion only after exact executor
/// retirement succeeds.
///
/// The writer is authenticated against `frame` before staging. Sampling only
/// appends while the writer stays open; executor retirement must then succeed
/// before `submit` publishes the completion. Transport delivery remains the
/// caller's responsibility. Once a ready call enters, any authentication,
/// staging, launch, sampling, finish, or submit failure permanently transitions
/// the owner to `WorkerFaulted` and best-effort aborts the writer. Faulted owners
/// reject readiness and further execution but retain deterministic close
/// authority.
pub fn DeviceWorkerOwner::execute_frame(
  self : DeviceWorkerOwner,
  frame : @worker_wire.ValidatedPlanFrame,
  writer : @worker_wire.CompletionFrameWriter,
) -> @worker_wire.ValidatedCompletionFrame raise DeviceWorkerError {
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

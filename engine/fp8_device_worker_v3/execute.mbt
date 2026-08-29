///|
fn[T] fail_execution_v3(
  owner : Fp8DeviceWorkerOwnerV3,
  writer : @worker_wire.CompletionFrameWriter,
  primary : Fp8DeviceWorkerV3Error,
) -> T raise Fp8DeviceWorkerV3Error {
  owner.lifecycle = WorkerFaulted
  try writer.abort() catch {
    abort_error => raise CompletionAbortFailed(primary~, abort=abort_error)
  } noraise {
    _ => raise primary
  }
}

///|
/// Executes one frame; scale poisoning or any failure permanently fault-stops.
pub fn Fp8DeviceWorkerOwnerV3::execute_frame(
  self : Self,
  frame : @worker_wire.ValidatedPlanFrame,
  writer : @worker_wire.CompletionFrameWriter,
) -> @worker_wire.ValidatedCompletionFrame raise Fp8DeviceWorkerV3Error {
  if self.lifecycle != WorkerReady {
    raise InvalidLifecycle(self.lifecycle)
  }
  guard self.resources.executor is Some(Complete(executor)) else {
    raise Invalid(ResourceState)
  }
  writer.validate_plan(frame) catch {
    error => return fail_execution_v3(self, writer, Wire(error))
  }
  let staged = executor.stage_frame(frame) catch {
    error => return fail_execution_v3(self, writer, Executor(error))
  }
  let executed = executor.execute(staged) catch {
    error => return fail_execution_v3(self, writer, Executor(error))
  }
  executor.sample_completion_frame(executed, frame, writer) catch {
    error => return fail_execution_v3(self, writer, Executor(error))
  }
  executor.finish(executed) catch {
    error => return fail_execution_v3(self, writer, Executor(error))
  }
  writer.submit() catch {
    error => return fail_execution_v3(self, writer, Wire(error))
  }
}

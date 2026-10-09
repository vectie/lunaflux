///|
pub fn DeviceWorkerOwner::streaming_command(
  self : DeviceWorkerOwner,
  command : @streaming_ir.Command,
) -> @streaming_ir.Reply raise DeviceWorkerError {
  if self.lifecycle != WorkerReady {
    raise InvalidLifecycle(self.lifecycle)
  }
  let executor = ready_executor(self.resources)
  executor.streaming_command(command) catch {
    error => {
      self.lifecycle = WorkerFaulted
      raise Executor(error)
    }
  }
}

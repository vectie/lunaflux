///|
pub fn Fp8DeviceWorkerOwnerV3::streaming_command(
  self : Fp8DeviceWorkerOwnerV3,
  command : @streaming_ir.Command,
) -> @streaming_ir.Reply raise Fp8DeviceWorkerV3Error {
  if self.lifecycle != WorkerReady {
    raise InvalidLifecycle(self.lifecycle)
  }
  guard self.resources.executor is Some(Complete(executor)) else {
    raise Invalid(ResourceState)
  }
  executor.streaming_command(command) catch {
    error => {
      self.lifecycle = WorkerFaulted
      raise Executor(error)
    }
  }
}

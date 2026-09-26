///|
pub fn DeviceWorkerBootstrapOwner::streaming_command(
  self : DeviceWorkerBootstrapOwner,
  command : @streaming_ir.Command,
) -> @streaming_ir.Reply raise DeviceWorkerBootstrapError {
  match self.worker {
    Legacy(owner) =>
      owner.streaming_command(command) catch {
        error => raise Worker(error)
      }
    _ => raise Invalid(NumericCapability)
  }
}

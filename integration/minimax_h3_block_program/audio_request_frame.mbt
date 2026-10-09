///|
/// Synchronous stage adapter for the joint pipeline. It waits once after the
/// complete ordered list, never between individual audio kernels.
pub fn AudioDecodeRequest::execute(
  self : Self,
) -> Unit raise @device.DeviceError {
  self.submit()
  guard self.queue is Some(queue) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorWait)
  }
  self.state.phase = Failed
  queue.wait_completion()
  self.state.phase = Complete
}

///|
pub impl @joint_diffusion_execution.PreparedFrame for AudioDecodeRequest with fn execute(
  self,
) {
  self.execute()
}

///|
pub impl @joint_diffusion_execution.PreparedFrame for AudioDecodeRequest with fn drain(
  self,
) {
  self.cancel()
}

///|
pub impl @joint_diffusion_execution.PreparedFrame for AudioDecodeRequest with fn release(
  self,
) {
  self.close()
}

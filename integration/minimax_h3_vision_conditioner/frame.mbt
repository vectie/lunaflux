///|
/// One event wait at the whole-tower boundary; no inter-kernel host barriers.
pub fn VisionRequest::wait(self : Self) -> Unit raise @device.DeviceError {
  guard self.phase == Submitted && self.queue is Some(queue) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorWait)
  }
  self.phase = Failed
  queue.wait_completion()
  self.phase = Complete
}

///|
pub fn VisionRequest::execute(self : Self) -> Unit raise @device.DeviceError {
  self.submit()
  self.wait()
}

///|
/// Startup-only address binding for a downstream prepared frame. The consumer
/// must not execute until this frame has completed; it retains allocation
/// leases and must close before this request. This does not expose valid data.
pub fn[T] VisionRequest::with_destination(
  self : Self,
  consume : (@device.Allocation, @vision.VisionPlan) -> T raise @device.DeviceError,
) -> T raise @device.DeviceError {
  guard self.phase == Prepared && self.allocations.length() == 3 else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  consume(self.allocations[1], self.source.plan)
}

///|
pub impl @joint_diffusion_execution.PreparedFrame for VisionRequest with fn execute(
  self,
) {
  self.execute()
}

///|
pub impl @joint_diffusion_execution.PreparedFrame for VisionRequest with fn drain(
  self,
) {
  self.cancel()
}

///|
pub impl @joint_diffusion_execution.PreparedFrame for VisionRequest with fn release(
  self,
) {
  self.close()
}

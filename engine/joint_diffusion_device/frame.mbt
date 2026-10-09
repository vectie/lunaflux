///|
/// Private narrow device adapter: queue leases retain stream, functions and
/// allocation regions. Their external owners can release them after all frame
/// queues close. No first-party native ABI or request-path compilation is added.
pub struct KernelFrame {
  priv queue : @device.OrderedKernelExecutor
}

///|
/// Startup-only. Each frame's ordered list must contain the exact prepared
/// denoiser launches followed by video and audio RF updates. Step-dependent
/// arguments are immutable; buffers can be shared through device queue leases.
pub fn KernelFrame::new(
  context : @device.Context,
  stream : @device.Stream,
  launches : ArrayView[@device.OrderedKernelLaunch],
  policy? : @device.OrderedKernelExecutionPolicy = OrderedKernelEagerOnly,
) -> Self raise @device.DeviceError {
  { queue: context.create_ordered_kernel_executor(stream, launches, policy~), }
}

///|
pub impl @joint_diffusion_execution.PreparedFrame for KernelFrame with fn execute(
  self,
) {
  match self.queue.execution_mode() {
    OrderedKernelCaptured => self.queue.launch_captured()
    OrderedKernelEager =>
      for index in 0..<self.queue.kernel_count() {
        self.queue.enqueue(index)
      }
  }
  self.queue.record_completion()
  self.queue.wait_completion()
}

///|
pub impl @joint_diffusion_execution.PreparedFrame for KernelFrame with fn drain(
  self,
) {
  self.queue.abort()
}

///|
pub impl @joint_diffusion_execution.PreparedFrame for KernelFrame with fn release(
  self,
) {
  self.queue.close()
}

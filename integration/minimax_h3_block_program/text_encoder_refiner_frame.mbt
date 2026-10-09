///|
pub impl @joint_diffusion_execution.PreparedFrame for TextConditioningProgram with fn execute(
  self,
) {
  self.execute()
}

///|
pub impl @joint_diffusion_execution.PreparedFrame for TextConditioningProgram with fn drain(
  self,
) {
  self.phase = Failed
  if self.queue is Some(queue) {
    queue.abort()
  }
}

///|
pub impl @joint_diffusion_execution.PreparedFrame for TextConditioningProgram with fn release(
  self,
) {
  self.close()
}

///|
/// Destination is bound, not initialized. A dependent denoising/conditioning
/// queue must only run after this refiner executes successfully to completion.
pub fn[T] TextConditioningProgram::with_destination(
  self : Self,
  consume : (@device.Allocation, Int64) -> T raise @device.DeviceError,
) -> T raise @device.DeviceError {
  guard self.phase == Prepared && self.output is Some(output) else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  consume(output, 0L)
}

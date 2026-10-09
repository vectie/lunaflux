///|
pub impl @joint_diffusion_execution.PreparedFrame for ConditioningProgram with fn execute(
  self,
) {
  self.execute()
}

///|
pub impl @joint_diffusion_execution.PreparedFrame for ConditioningProgram with fn drain(
  self,
) {
  self.phase = Failed
  if self.queue is Some(queue) {
    queue.abort()
  }
}

///|
pub impl @joint_diffusion_execution.PreparedFrame for ConditioningProgram with fn release(
  self,
) {
  self.close()
}

///|
/// Startup binding only. Prepared contents are uninitialized until this stage
/// completes; callers must order every prediction after conditioning execution.
pub fn[T] ConditioningProgram::with_destination(
  self : Self,
  consume : (@device.Allocation, Int64) -> T raise @device.DeviceError,
) -> T raise @device.DeviceError {
  guard self.phase is (Prepared | Complete) && self.output is Some(output) else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  consume(output, 0L)
}

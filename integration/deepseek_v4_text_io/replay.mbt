///|
/// Offline borrowed semantic ports from the actual prepared learned head.
/// Retire the borrowing queue before capture; preserve the literal input IDs,
/// absolute positions and selected-row mapping outside the compiler.
pub fn DeepSeekTextIo::replay_buffers(
  self : Self,
) -> Array[(@precision.NumericBoundary, @device.Allocation)] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(DeviceCopyCreate)
  }
  self.frame.replay_buffers()
}

///|
pub fn DeepSeekTextIo::projection_suffix(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.frame.projection_suffix()
}

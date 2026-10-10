///|
/// Startup-only borrowed state enumeration for a prepared rollback transaction.
/// Includes ring payload and frontier; append metadata/output are recomputed.
pub fn WindowSharedKvFrame::persistent_state(
  self : Self,
) -> Array[@device.Allocation] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(DeviceCopyCreate)
  }
  [self.buffers[0], self.buffers[1]]
}

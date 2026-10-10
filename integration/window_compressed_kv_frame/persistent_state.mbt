///|
/// Startup-only borrowed window payload/frontier. The separately owned learned
/// compressed cache and partial pooling groups must also enter the transaction.
pub fn WindowCompressedKvFrame::persistent_state(
  self : Self,
) -> Array[@device.Allocation] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(DeviceCopyCreate)
  }
  [self.buffers[0], self.buffers[1]]
}

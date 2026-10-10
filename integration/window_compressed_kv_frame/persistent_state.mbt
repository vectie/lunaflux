///|
/// Pure startup footprint of the retained window, excluding append/output
/// scratch and the separately owned learned compressed cache.
pub fn WindowCompressedKvFrame::persistent_state_bytes(self : Self) -> Int64 {
  let sizes = self.source.plan().window().buffer_bytes()
  sizes[0] + sizes[1]
}

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

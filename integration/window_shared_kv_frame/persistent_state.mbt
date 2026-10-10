///|
/// Pure startup footprint of ring payload and frontier only. The full frame
/// state_bytes also includes recomputable append/output scratch.
pub fn WindowSharedKvFrame::persistent_state_bytes(self : Self) -> Int64 {
  let sizes = self.source.plan().buffer_bytes()
  sizes[0] + sizes[1]
}

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

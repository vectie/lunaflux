///|
/// Pure startup footprint, excluding per-invocation append/output scratch.
pub fn RetainedRowsFrame::persistent_state_bytes(self : Self) -> Int64 {
  self.source.plan().state_bytes().fold(init=0L, (total, size) => total + size)
}

///|
/// Retained values, append frontier and consumer-visible counts commit together.
/// The per-invocation append descriptor remains recomputable scratch.
pub fn RetainedRowsFrame::persistent_state(
  self : Self,
) -> Array[@device.Allocation] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(DeviceCopyCreate)
  }
  [self.buffers[0], self.buffers[1], self.buffers[2]]
}

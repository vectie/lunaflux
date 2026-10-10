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

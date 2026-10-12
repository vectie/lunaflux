///|
/// Offline borrowed ports only. The caller must wait for the named producer,
/// capture live extents from append, and finish copies before resuming/reusing
/// this frame. This performs no copy, synchronization, evaluation or hashing.
/// Ordinary serving never calls it. Previous state uses persistent_state().
pub fn LearnedGatedPoolFrame::replay_buffers(
  self : Self,
) -> Array[(@precision.NumericBoundary, @device.Allocation)] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(DeviceCopyCreate)
  }
  [
    ..self.source
    .plan()
    .replay_boundaries()
    .map(boundary => (boundary, self.buffers[boundary.buffer])),
  ]
}

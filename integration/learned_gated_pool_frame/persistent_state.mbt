///|
/// Pure startup footprint of exactly the allocations borrowed by a snapshot.
pub fn LearnedGatedPoolFrame::persistent_state_bytes(self : Self) -> Int64 {
  self.source.plan().state_bytes().fold(init=0L, (total, size) => total + size)
}

///|
/// Partial-group numerators, gates and history survive across calls. All three
/// are restored even when a tentative row completes or overwrites a group.
pub fn LearnedGatedPoolFrame::persistent_state(
  self : Self,
) -> Array[@device.Allocation] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(DeviceCopyCreate)
  }
  [self.buffers[0], self.buffers[1], self.buffers[2]]
}

///|
/// Ring payload and committed frontier/error only. Descriptors and attention
/// output are recomputed scratch and do not belong to an undo transaction.
pub fn CommittedDraftKvFrame::persistent_state_bytes(self : Self) -> Int64 {
  let sizes = self.prime.plan().buffer_bytes()
  sizes[0] + sizes[1]
}

///|
/// Startup enumeration. The enclosing queue must be idle before preparing a
/// transaction and retire before restoration; the frame owns no completion.
pub fn CommittedDraftKvFrame::persistent_state(
  self : Self,
) -> Array[@device.Allocation] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(DeviceCopyCreate)
  }
  [self.buffers[0], self.buffers[1]]
}

///|
/// Offline borrowed intermediates; caller must retire its enclosing queue.
/// This visitor does not execute, allocate, copy, or alter a serving launch.
pub fn GroupedAttentionOutputFrame::visit_retired_outputs(
  self : Self,
  observe : (String, @device.Allocation, Int) -> Unit raise,
) -> Unit raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorPoll)
  }
  let (_, _, _, rows) = self.source.plan().geometry()
  let bytes = self.source.plan().frame_bytes()
  observe(
    "attention-inverse",
    self.buffers[0],
    (bytes[0] / rows.to_int64() / 2L).to_int(),
  )
  observe(
    "attention-output-a",
    self.buffers[1],
    (bytes[1] / rows.to_int64() / 2L).to_int(),
  )
}

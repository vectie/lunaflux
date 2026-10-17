///|
/// Fixed-capacity rank-major candidates, gathered immediately before the
/// global greedy join. Borrowed through queue retirement; no host readback.
pub fn TextIoFrame::vocabulary_boundary(
  self : Self,
) -> (Int, @device.Allocation, @device.Allocation, Int, Int)? raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  match self.source.vocabulary_shard() {
    None => None
    Some(shard) =>
      Some(
        (
          3,
          self.buffers[2],
          self.buffers[3],
          shard.candidate_width(),
          shard.group().1,
        ),
      )
  }
}

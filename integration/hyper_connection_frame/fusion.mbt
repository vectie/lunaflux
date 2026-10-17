///|
/// Bind a proven adjacent publication -> square edge. Both frame owners and
/// their borrowed input/output allocations must outlive the composed queue.
/// The caller replaces precisely those two launches, not the branch or its
/// collective. Unmatched aliases/laws are rejected at startup.
pub fn HyperConnectionFrame::fuse_post_square(
  self : Self,
  next : Self,
  aot : @device.Module,
) -> @device.OrderedKernelLaunch raise {
  guard self.status is Prepared &&
    next.status is Prepared &&
    next.source.supports_post_square() &&
    self.ports[0].same_storage(next.ports[0]) &&
    self.ports[2].same_storage(next.ports[1]) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let plan = @physical.PostSquareFusion::new(
    self.source.plan(),
    next.source.plan(),
  )
  let (streams, hidden, rows, workers) = plan.geometry()
  let bytes = rows.to_int64() * streams.to_int64() * hidden.to_int64() * 2L
  let sizes = self.source.plan().frame_bytes()
  guard next.source.control_schedule() is Some(schedule) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  next.launch(aot, 6, rows, workers, [
    region(self.ports[0], 20L, 4L),
    region(self.buffers[0], sizes[0], 4L),
    region(self.buffers[2], sizes[2], 4L),
    region(self.ports[1], bytes, 2L),
    region(self.buffers[5], sizes[5], 2L),
    region(self.ports[2], bytes, 2L),
    region(next.buffers[6], schedule.workspace_bytes(), 4L),
  ])
}

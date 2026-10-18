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

///|
pub fn HyperConnectionFrame::fuse_control_tail(
  self : Self,
  aot : @device.Module,
) -> @device.OrderedKernelLaunch raise {
  guard self.status is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let plan = @physical.ControlTailFusion::new(self.source.plan())
  let (rows, workers, _) = plan.geometry()
  let arguments = self.controls.copy()
  arguments.push(
    region(self.buffers[2], self.source.plan().frame_bytes()[2], 4L),
  )
  self.launch(aot, 7, rows, workers, arguments)
}

///|
/// Delay the producer publication to the successor's control projection.
/// The immutable schedule preserves every numerical publication and owns only
/// private intermediates. All allocations remain borrowed from the two frames.
pub fn HyperConnectionFrame::fuse_deferred_post_pre(
  self : Self,
  next : Self,
  aot : @device.Module,
) -> @device.OrderedKernelLaunch raise {
  guard self.status is Prepared &&
    next.status is Prepared &&
    self.ports[0].same_storage(next.ports[0]) &&
    self.ports[2].same_storage(next.ports[1]) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let (streams, hidden, rows, workers) = @physical.PostSquareFusion::new(
    self.source.plan(),
    next.source.plan(),
  ).geometry()
  let dispatch = next.source.deferred_dispatch()
  guard next.source.control_schedule() is Some(schedule) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let bytes = rows.to_int64() * streams.to_int64() * hidden.to_int64() * 2L
  let sizes = self.source.plan().frame_bytes()
  next.launch(aot, 8, dispatch.workgroups(), workers, [
    region(self.ports[0], 20L, 4L),
    region(self.buffers[0], sizes[0], 4L),
    region(self.buffers[2], sizes[2], 4L),
    region(self.ports[1], bytes, 2L),
    region(self.buffers[5], sizes[5], 2L),
    region(self.ports[2], bytes, 2L),
    region(next.ports[3], next.source.plan().function_weight_bytes(), 2L),
    region(next.buffers[6], schedule.workspace_bytes(), 4L),
  ])
}

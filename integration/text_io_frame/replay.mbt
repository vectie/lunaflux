///|
/// Borrow semantic ports after the producing queue retires and before reuse.
/// No synchronization/copy or comparison happens here. Selected rows are local
/// frame rows, NOT absolute token positions. The offline adapter supplies that
/// mapping and validates count/range fields before reading live prefixes.
pub fn TextIoFrame::replay_buffers(
  self : Self,
) -> Array[(@precision.NumericBoundary, @device.Allocation)] raise {
  guard self.state is Prepared && self.boundary.has_egress() else {
    raise @device.DeviceError::InvalidArgument(DeviceCopyCreate)
  }
  let ports = [..self.buffers, ..self.replay_inputs]
  [
    ..self.source
    .plan()
    .replay_boundaries()
    .map(port => (port, ports[port.buffer])),
  ]
}

///|
/// Normalization and raw model head only. Sampling is a separate consumer, not
/// part of teacher-forced logits. The ordinary three-stage suffix is unchanged.
pub fn TextIoFrame::projection_suffix(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared && self.boundary.has_egress() else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.after[:2]
}

///|
pub fn LearnedTextIoFrame::replay_buffers(
  self : Self,
) -> Array[(@precision.NumericBoundary, @device.Allocation)] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(DeviceCopyCreate)
  }
  let ports = [
    self.buffers[0],
    ..self.text.replay_buffers().map(entry => entry.1),
  ]
  [..self.plan.replay_boundaries().map(port => (port, ports[port.buffer]))]
}

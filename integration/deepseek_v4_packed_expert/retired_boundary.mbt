///|
/// Offline observer of borrowed branch ports, after enclosing queue retirement.
/// No readback or diagnostic work is introduced into ordinary submit/poll.
pub fn[T] DeepSeekFfnEnvelope::with_branch_ports(
  self : Self,
  observe : (@device.Allocation, @device.Allocation) -> T raise,
) -> T raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorPoll)
  }
  observe(self.hyper.branch_input(), self.hyper.branch_output())
}

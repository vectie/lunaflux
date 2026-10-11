///|
/// Offline observer of borrowed branch ports. The enclosing executor must be
/// retired before calling; this adds no launch, allocation or serving scan.
pub fn[T] DeepSeekAttentionEnvelope::with_branch_ports(
  self : Self,
  observe : (@device.Allocation, @device.Allocation) -> T raise,
) -> T raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorPoll)
  }
  observe(self.hyper.branch_input(), self.hyper.branch_output())
}

///|
/// Startup-bound addresses for an offline observer. Caller must retire the
/// enclosing executor before reading and finish before resubmission. The array
/// owns no new device allocation and installs no observer in the launch plan.
pub fn MoeProgram::borrow_retired_frontiers(
  self : Self,
) -> ArrayView[(String, @device.Allocation)] raise @device.DeviceError {
  guard !self.closed && !self.frontiers.is_empty() else {
    raise @device.DeviceError::InvalidArgument(StreamingTransfer)
  }
  self.frontiers[:]
}

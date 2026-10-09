///|
/// Host-side request ownership, confined to the preparation/control thread.
/// A lease lasts until queue closure, including failure and cancellation.
struct AudioDecoderLease {
  mut held : Bool
}

///|
fn AudioDecoderLease::acquire(self : Self) -> Unit raise @device.DeviceError {
  guard !self.held else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.held = true
}

///|
fn AudioDecoderLease::require_idle(
  self : Self,
) -> Unit raise @device.DeviceError {
  guard !self.held else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
}

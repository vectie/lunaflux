///|
priv enum AudioRequestPhase {
  Empty
  Prepared
  Submitted
  Complete
  Cancelled
  Failed
  Closed
} derive(Eq)

///|
struct AudioRequestState {
  mut phase : AudioRequestPhase
}

///|
fn AudioRequestState::submit(
  self : Self,
  enqueue : () -> Unit raise @device.DeviceError,
) -> Unit raise @device.DeviceError {
  guard self.phase == Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue)
  }
  self.phase = Failed
  enqueue()
  self.phase = Submitted
}

///|
fn AudioRequestState::poll(
  self : Self,
  completed : () -> Bool raise @device.DeviceError,
) -> Bool raise @device.DeviceError {
  if self.phase == Complete {
    return true
  }
  guard self.phase == Submitted else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue)
  }
  self.phase = Failed
  let done = completed()
  self.phase = if done { Complete } else { Submitted }
  done
}

///|
fn AudioRequestState::cancel(
  self : Self,
  drain : () -> Unit raise @device.DeviceError,
) -> Unit raise @device.DeviceError {
  if self.phase == Closed || self.phase == Cancelled {
    return
  }
  self.phase = Failed
  drain()
  self.phase = Cancelled
}

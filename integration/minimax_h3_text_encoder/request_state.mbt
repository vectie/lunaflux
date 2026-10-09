///|
fn submit_once(
  phase : Phase,
  run : () -> Unit raise @device.DeviceError,
) -> Phase raise @device.DeviceError {
  guard phase == Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue)
  }
  run()
  Submitted
}

///|
fn poll_once(
  phase : Phase,
  run : () -> Bool raise @device.DeviceError,
) -> (Phase, Bool) raise @device.DeviceError {
  if phase == Complete {
    return (Complete, true)
  }
  guard phase == Submitted else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorPoll)
  }
  let done = run()
  (if done { Complete } else { Submitted }, done)
}

///|
fn deepstack_stage(layer : Int) -> Int? {
  if layer >= 0 && layer < 3 {
    Some(layer + 1)
  } else {
    None
  }
}

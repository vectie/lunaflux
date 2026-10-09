///|
test "request submission is one shot and completion does not rerun GPU callback" {
  let calls = Ref(0)
  let submitted = submit_once(Prepared, () => calls.val += 1)
  assert_true(submitted == Submitted)
  let (waiting, done) = poll_once(submitted, () => {
    calls.val += 1
    false
  })
  assert_true(waiting == Submitted && !done)
  let (complete, done) = poll_once(waiting, () => {
    calls.val += 1
    true
  })
  assert_true(complete == Complete && done)
  poll_once(complete, () => {
    calls.val += 100
    false
  })
  |> ignore
  assert_eq(calls.val, 3)
  let forbidden : Array[Phase] = [
    Empty,
    Submitted,
    Complete,
    Failed,
    Cancelled,
    Closed,
  ]
  for phase in forbidden {
    try submit_once(phase, () => calls.val += 100) catch {
      @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue) => ()
      _ => fail("wrong state error")
    } noraise {
      _ => fail("request replayed")
    }
  }
  assert_eq(calls.val, 3)
}

///|
test "GPU failure propagates without marking a completed request" {
  try
    submit_once(Prepared, () => {
      raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
    })
  catch {
    @device.DeviceError::InvalidArgument(AotKernelLaunch) => ()
    _ => fail("failure lost")
  } noraise {
    _ => fail("failed submit completed")
  }
  try
    poll_once(Submitted, () => {
      raise @device.DeviceError::InvalidArgument(OrderedExecutorPoll)
    })
  catch {
    @device.DeviceError::InvalidArgument(OrderedExecutorPoll) => ()
    _ => fail("failure lost")
  } noraise {
    _ => fail("failed poll completed")
  }
}

///|
test "deepstack outputs inject after language layers zero one two only" {
  assert_eq(deepstack_stage(0), Some(1))
  assert_eq(deepstack_stage(1), Some(2))
  assert_eq(deepstack_stage(2), Some(3))
  for layer in [3, 8, 16, 24, 49, -1] {
    assert_eq(deepstack_stage(layer), None)
  }
}

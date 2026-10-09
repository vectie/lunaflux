///|
test "text conditioning storage uses checked exact BF16 extent" {
  assert_eq(text_conditioning_bytes(3, 5376, 32256L), 32256L)
  for
    (rows, width, budget) in [
      (3, 5376, 32255L),
      (0, 5376, 100000L),
      (3, 0, 100000L),
      (3, 5376, -1L),
      (2147483647, 2147483647, 100000L),
    ] {
    try text_conditioning_bytes(rows, width, budget) catch {
      @device.DeviceError::InvalidArgument(AllocationCreate) => ()
      _ => fail("wrong extent error")
    } noraise {
      _ => fail("invalid extent accepted")
    }
  }
}

///|
test "text conditioning executes only prepared phase and propagates failure" {
  let calls = Ref(0)
  let complete = execute_conditioning_once(Prepared, () => calls.val += 1)
  assert_true(complete == Complete)
  assert_eq(calls.val, 1)
  let phases : Array[ConditioningPhase] = [Empty, Complete, Failed]
  for phase in phases {
    try execute_conditioning_once(phase, () => calls.val += 1) catch {
      @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue) => ()
      _ => fail("wrong phase error")
    } noraise {
      _ => fail("re-executed conditioning")
    }
  }
  assert_eq(calls.val, 1)
  try
    execute_conditioning_once(Prepared, () => {
      raise @device.DeviceError::InvalidArgument(OrderedExecutorWait)
    })
  catch {
    @device.DeviceError::InvalidArgument(OrderedExecutorWait) => ()
    _ => fail("lost execution error")
  } noraise {
    _ => fail("failed execution completed")
  }
}

///|
test "audio request completion is observed once and submission cannot replay" {
  let state : AudioRequestState = { phase: Prepared, }
  let mut submissions = 0
  state.submit(() => submissions += 1)
  assert_true(state.phase == Submitted)
  assert_false(state.poll(() => false))
  assert_true(state.phase == Submitted)
  assert_true(state.poll(() => true))
  assert_true(state.phase == Complete)
  let mut polls = 0
  assert_true(
    state.poll(() => {
      polls += 1
      false
    }),
  )
  assert_eq(polls, 0)
  try state.submit(() => submissions += 1) catch {
    _ => ()
  } noraise {
    _ => fail("completed request replayed")
  }
  assert_eq(submissions, 1)
}

///|
test "audio partial submission and poll failures require drain without retry" {
  let state : AudioRequestState = { phase: Prepared, }
  try
    state.submit(() => {
      raise @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue)
    })
  catch {
    _ => ()
  } noraise {
    _ => fail("submit failure lost")
  }
  assert_true(state.phase == Failed)
  let mut drained = 0
  state.cancel(() => drained += 1)
  assert_true(state.phase == Cancelled)
  state.cancel(() => drained += 1)
  assert_eq(drained, 1)
  let pending : AudioRequestState = { phase: Submitted, }
  try
    pending.poll(() => {
      raise @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue)
    })
  catch {
    _ => ()
  } noraise {
    _ => fail("poll failure lost")
  }
  assert_true(pending.phase == Failed)
  try
    pending.cancel(() => {
      raise @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue)
    })
  catch {
    _ => ()
  } noraise {
    _ => fail("drain failure lost")
  }
  assert_true(pending.phase == Failed)
  pending.cancel(() => ())
  assert_true(pending.phase == Cancelled)
}

///|
test "audio invalid request phases do not submit or poll device work" {
  let phases : Array[AudioRequestPhase] = [
    Empty,
    Submitted,
    Complete,
    Cancelled,
    Failed,
    Closed,
  ]
  let mut calls = 0
  for phase in phases {
    let state : AudioRequestState = { phase, }
    try state.submit(() => calls += 1) catch {
      _ => ()
    } noraise {
      _ => fail("invalid submit accepted")
    }
  }
  assert_eq(calls, 0)
}

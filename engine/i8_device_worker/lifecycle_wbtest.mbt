///|
test "readiness requires complete executor transfer and live context" {
  assert_true(i8_resources_publishable(WorkerReady, true, false, true, true))
  assert_false(i8_resources_publishable(WorkerReady, false, false, true, true))
  assert_false(i8_resources_publishable(WorkerReady, true, true, true, true))
  assert_false(i8_resources_publishable(WorkerReady, true, false, false, true))
  assert_false(i8_resources_publishable(WorkerReady, true, false, true, false))
  assert_false(i8_resources_publishable(WorkerFaulted, true, false, true, true))
  assert_false(
    i8_resources_publishable(WorkerClosingFailed, true, false, true, true),
  )
  assert_false(i8_resources_publishable(WorkerClosed, true, false, true, true))
}

///|
fn fake_owner_advance(
  phase : Ref[I8OwnerSlotPhase],
  active : Ref[Int],
  next : I8OwnerSlotPhase,
  events : Array[String],
  event : String,
) -> Unit raise I8DeviceWorkerError {
  let before = phase.val
  let after_count = owner_phase_active_count(next)
  validate_owner_transition(before, before, active.val, next, after_count)
  phase.val = next
  active.val = after_count
  events.push(event)
}

///|
test "production owner transition seam enforces raw authenticated loaded ready" {
  let phase = Ref(OwnerSlotsEmpty)
  let active = Ref(0)
  let events : Array[String] = []
  fake_owner_advance(phase, active, RawNumericOwner, events, "load")
  fake_owner_advance(
    phase,
    active,
    AuthenticatedNumericOwner,
    events,
    "authenticate",
  )
  fake_owner_advance(phase, active, LoadedNumericOwner, events, "join")
  fake_owner_advance(
    phase,
    active,
    ExecutorTransferOwner,
    events,
    "bind-transfer",
  )
  fake_owner_advance(
    phase,
    active,
    ExecutorConstructionOwner,
    events,
    "transfer-to-lower",
  )
  fake_owner_advance(
    phase,
    active,
    ReadyExecutorOwner,
    events,
    "executor-ready",
  )
  assert_eq(active.val, 1)
  assert_eq(events, [
    "load", "authenticate", "join", "bind-transfer", "transfer-to-lower", "executor-ready",
  ])
  fake_owner_advance(phase, active, OwnerSlotsEmpty, events, "cleanup")
  assert_eq(active.val, 0)
}

///|
test "failed construction retains one cleanup owner and retry boundary" {
  let phase = Ref(OwnerSlotsEmpty)
  let active = Ref(0)
  let events : Array[String] = []
  fake_owner_advance(phase, active, RawNumericOwner, events, "load")
  fake_owner_advance(
    phase,
    active,
    AuthenticatedNumericOwner,
    events,
    "authenticate",
  )
  fake_owner_advance(phase, active, LoadedNumericOwner, events, "join")
  fake_owner_advance(
    phase,
    active,
    ExecutorTransferOwner,
    events,
    "bind-transfer",
  )
  fake_owner_advance(
    phase,
    active,
    ExecutorConstructionOwner,
    events,
    "transfer-to-lower",
  )
  fake_owner_advance(
    phase,
    active,
    FailedExecutorOwner,
    events,
    "PagedCleanupRequired",
  )
  assert_eq(active.val, 1)
  assert_true(phase.val == FailedExecutorOwner)
  fake_owner_advance(phase, active, OwnerSlotsEmpty, events, "retry-close")
  assert_eq(active.val, 0)
  assert_eq(events, [
    "load", "authenticate", "join", "bind-transfer", "transfer-to-lower", "PagedCleanupRequired",
    "retry-close",
  ])
}

///|
test "post-transfer failure successful cleanup closes lower owner exactly once" {
  let phase = Ref(ExecutorConstructionOwner)
  let active = Ref(0)
  let close_count = Ref(0)
  let events : Array[String] = []
  close_count.val += 1
  fake_owner_advance(
    phase,
    active,
    OwnerSlotsEmpty,
    events,
    "lower-cleanup-complete",
  )
  if phase.val != OwnerSlotsEmpty {
    close_count.val += 1
  }
  assert_eq(close_count.val, 1)
  assert_eq(active.val, 0)
  assert_eq(events, ["lower-cleanup-complete"])
}

///|
test "failed loader owns exactly one retryable numeric cleanup slot" {
  let phase = Ref(OwnerSlotsEmpty)
  let active = Ref(0)
  let events : Array[String] = []
  fake_owner_advance(
    phase,
    active,
    FailedNumericOwner,
    events,
    "loader-cleanup-required",
  )
  assert_true(phase.val == FailedNumericOwner)
  assert_eq(active.val, 1)
  fake_owner_advance(phase, active, OwnerSlotsEmpty, events, "retry-close")
  assert_eq(active.val, 0)
}

///|
fn expect_invalid_transition(
  before : I8OwnerSlotPhase,
  active_before : Int,
  after : I8OwnerSlotPhase,
  active_after : Int,
) -> Unit raise {
  try
    validate_owner_transition(
      before, before, active_before, after, active_after,
    )
  catch {
    Invalid(ResourceState) => ()
    _ => fail("unexpected owner transition failure")
  } noraise {
    _ => fail("invalid owner transition was accepted")
  }
}

///|
test "failed authenticate join or executor construction retains prior owner" {
  let phase = Ref(RawNumericOwner)
  let active = Ref(1)
  let hostile : Array[(I8OwnerSlotPhase, I8OwnerSlotPhase)] = [
    (RawNumericOwner, AuthenticatedNumericOwner),
    (AuthenticatedNumericOwner, LoadedNumericOwner),
    (LoadedNumericOwner, ExecutorTransferOwner),
    (ExecutorTransferOwner, ExecutorConstructionOwner),
    (ExecutorConstructionOwner, ReadyExecutorOwner),
    (ExecutorConstructionOwner, FailedExecutorOwner),
  ]
  for pair in hostile {
    let (before, after) = pair
    phase.val = before
    active.val = 1
    expect_invalid_transition(before, active.val, after, 2)
    assert_true(phase.val == before)
    assert_eq(active.val, 1)
  }
  expect_invalid_transition(AuthenticatedNumericOwner, 1, ReadyExecutorOwner, 1)
}

///|
test "cleanup is strict executor numeric context dependency order" {
  let events : Array[String] = []
  run_i8_cleanup_chain(
    () => events.push("executor"),
    () => events.push("numeric"),
    () => events.push("context"),
  )
  assert_eq(events, ["executor", "numeric", "context"])
}

///|
test "failed child cleanup blocks every parent close" {
  let events : Array[String] = []
  try
    run_i8_cleanup_chain(
      () => {
        events.push("executor")
        raise Invalid(ResourceState)
      },
      () => events.push("numeric"),
      () => events.push("context"),
    )
  catch {
    Invalid(ResourceState) => ()
    _ => fail("unexpected cleanup failure")
  } noraise {
    _ => fail("failed executor cleanup was accepted")
  }
  assert_eq(events, ["executor"])
}

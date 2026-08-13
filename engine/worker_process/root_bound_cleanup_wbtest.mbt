///|
test "initial cleanup probes child and roots after double failure" {
  let events : Array[String] = []
  let state = attempt_root_bound_cleanup(
    true,
    true,
    () => {
      events.push("child")
      Some(Process(Transport(OperationFailed)))
    },
    () => {
      events.push("roots")
      Some(RootsBusy)
    },
    () => true,
    () => true,
  )
  assert_eq(events, ["child", "roots"])
  assert_true(state.child_live)
  assert_true(state.roots_live)
  assert_eq(state.child_failure, Some(Process(Transport(OperationFailed))))
  assert_eq(state.roots_failure, Some(RootsBusy))
}

///|
test "root Busy retains authority while consumed failure does not" {
  let busy = attempt_root_bound_cleanup(
    false,
    true,
    () => None,
    () => Some(RootsBusy),
    () => false,
    () => true,
  )
  assert_true(busy.roots_live)
  let consumed = attempt_root_bound_cleanup(
    false,
    true,
    () => None,
    () => Some(RootsCloseFailed),
    () => false,
    () => false,
  )
  assert_false(consumed.roots_live)
  assert_eq(consumed.roots_failure, Some(RootsCloseFailed))
}

///|
test "recovery close and replacement cleanup retry exact child authority" {
  let mut attempts = 0
  let first = attempt_root_bound_cleanup(
    true,
    false,
    () => {
      attempts += 1
      Some(Process(Transport(OperationFailed)))
    },
    () => None,
    () => true,
    () => false,
  )
  assert_true(first.child_live)
  let second = attempt_root_bound_cleanup(
    first.child_live,
    false,
    () => {
      attempts += 1
      None
    },
    () => None,
    () => false,
    () => false,
  )
  assert_false(second.child_live)
  assert_eq(attempts, 2)
  let replacement = attempt_root_bound_cleanup(
    true,
    false,
    () => None,
    () => None,
    () => false,
    () => false,
  )
  assert_false(replacement.child_live)
}

///|
test "consumed child close error preserves evidence without retry authority" {
  let consumed = attempt_root_bound_cleanup(
    true,
    false,
    () => Some(Process(Transport(OperationFailed))),
    () => None,
    () => false,
    () => false,
  )
  assert_false(consumed.child_live)
  try raise_cleanup_attempt(consumed) catch {
    RootBoundFailure(Process(Transport(OperationFailed))) => ()
    _ => fail("consumed child close evidence was misclassified")
  } noraise {
    _ => fail("consumed child close evidence was erased")
  }
}

///|
test "historical consumed evidence survives the other authority retry" {
  let historical_child = Some(Process(Transport(OperationFailed)))
  let first = attempt_root_bound_cleanup(
    true,
    true,
    () => historical_child,
    () => Some(RootsBusy),
    () => false,
    () => true,
  )
  assert_false(first.child_live)
  assert_true(first.roots_live)
  let second = attempt_root_bound_cleanup(
    first.child_live,
    first.roots_live,
    () => None,
    () => None,
    () => false,
    () => false,
  )
  let merged = {
    child_live: second.child_live,
    roots_live: second.roots_live,
    child_failure: match second.child_failure {
      Some(failure) => Some(failure)
      None => first.child_failure
    },
    roots_failure: second.roots_failure,
  }
  try raise_cleanup_attempt(merged) catch {
    RootBoundFailure(Process(Transport(OperationFailed))) => ()
    _ => fail("historical consumed failure changed during retry")
  } noraise {
    _ => fail("historical consumed failure disappeared during retry")
  }
}

///|
test "terminal cleanup attempts independent roots and preserves compound facts" {
  let events : Array[String] = []
  let state = attempt_root_bound_cleanup(
    true,
    true,
    () => {
      events.push("child")
      Some(Process(Transport(OperationFailed)))
    },
    () => {
      events.push("roots")
      Some(RootsCloseFailed)
    },
    () => true,
    () => false,
  )
  assert_eq(events, ["child", "roots"])
  assert_true(state.child_live)
  assert_false(state.roots_live)
  try raise_cleanup_attempt(state) catch {
    RootBoundCompoundFailure(
      Process(Transport(OperationFailed)),
      RootsCloseFailed
    ) => ()
    _ => fail("compound cleanup facts were not preserved")
  } noraise {
    _ => fail("compound cleanup failure was omitted")
  }
}

///|
test "cleanup error mapping distinguishes live Busy from consumed failure" {
  let busy = attempt_root_bound_cleanup(
    false,
    true,
    () => None,
    () => Some(RootsBusy),
    () => false,
    () => true,
  )
  try raise_cleanup_attempt(busy) catch {
    RootBoundFailure(RootsBusy) => ()
    _ => fail("live root cleanup did not report Busy")
  } noraise {
    _ => fail("live root authority was omitted")
  }
  let consumed = attempt_root_bound_cleanup(
    false,
    true,
    () => None,
    () => Some(RootsCloseFailed),
    () => false,
    () => false,
  )
  try raise_cleanup_attempt(consumed) catch {
    RootBoundFailure(RootsCloseFailed) => ()
    _ => fail("consumed close failure was mapped as live authority")
  } noraise {
    _ => fail("consumed close failure evidence was omitted")
  }
}

///|
test "child liveness probe is authoritative over close evidence" {
  let unexplained_live = attempt_root_bound_cleanup(
    true,
    false,
    () => None,
    () => None,
    () => true,
    () => false,
  )
  assert_true(unexplained_live.child_live)
  let consumed_failure = attempt_root_bound_cleanup(
    true,
    false,
    () => Some(Process(Transport(OperationFailed))),
    () => None,
    () => false,
    () => false,
  )
  assert_false(consumed_failure.child_live)
  assert_eq(
    consumed_failure.child_failure,
    Some(Process(Transport(OperationFailed))),
  )
}

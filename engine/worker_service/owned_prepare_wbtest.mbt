///|
fn owned_test_process_error() -> @worker_process.RootBoundWorkerProcessError {
  RootBoundFailure(RootsCloseFailed)
}

///|
fn owned_test_cleanup() -> FailedOwnedWorkerServicePreparation {
  {
    initial: None,
    ready_process: None,
    cause_kind: 3,
    service_cause: None,
    process_cause: Some(owned_test_process_error()),
    cleanup_process_failure: Some(owned_test_process_error()),
    closed: true,
    reported_consumed_failure: false,
  }
}

///|
test "owned preparation shell transfers only the selected owner once" {
  let cleanup = owned_test_cleanup()
  let prepared : OwnedWorkerServicePreparation = {
    service: None,
    cleanup,
    state: OwnedServiceCleanupRequired,
    consumed: false,
  }
  assert_eq(prepared.state(), OwnedServiceCleanupRequired)
  assert_eq(
    prepared.take_cleanup().cause(),
    OwnedProcessFailure(owned_test_process_error()),
  )
  try prepared.take_cleanup() catch {
    OwnedPreparationFailure(OwnedServiceFailure(Invalid(Lifecycle))) => ()
    _ => fail("unexpected duplicate cleanup extraction error")
  } noraise {
    _ => fail("cleanup authority extracted twice")
  }
  try prepared.take_ready() catch {
    OwnedPreparationFailure(OwnedServiceFailure(Invalid(Lifecycle))) => ()
    _ => fail("unexpected failed-service extraction error")
  } noraise {
    _ => fail("failed service escaped cleanup outcome")
  }
}

///|
test "consumed cleanup failure is reported once and cause persists" {
  let cleanup = owned_test_cleanup()
  let cause = cleanup.cause()
  try cleanup.retry_close() catch {
    OwnedPreparationFailure(
      OwnedProcessFailure(RootBoundFailure(RootsCloseFailed))
    ) => ()
    _ => fail("unexpected consumed cleanup evidence")
  } noraise {
    _ => fail("consumed cleanup evidence was dropped")
  }
  assert_eq(cleanup.cause(), cause)
  cleanup.retry_close()
  assert_true(cleanup.is_closed())
}

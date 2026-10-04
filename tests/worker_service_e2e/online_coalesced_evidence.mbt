///|
/// The reactor may commit a just-read completion in the same turn, but must
/// not submit a successor until its token/terminal publication is consumed.
fn run_online_coalesced_evidence(
  executable : Bytes,
  fixture_root? : String = "/private/tmp",
) -> Unit raise {
  let clock = @monotonic_clock.MonotonicClock::system()
  for cancelled in [false, true] {
    let lease = owned_online_lease(executable, clock, fixture_root~)
    lease.admit(online_request(701UL, clock.now_millis(), max_new_tokens=2))
    assert_true(lease.progress() is ServiceStarted(1UL))
    if cancelled {
      assert_eq(lease.reserve_cancel(), OnlineCancelReserved)
      assert_eq(lease.commit_cancel(), OnlineCancelCommitted)
      assert_false(lease.has_publication())
    }
    let mut attempts = 0
    while !lease.has_publication() && attempts < 1000000 {
      let status = lease.progress_status()
      if status is (OnlineWorkerOutputInvalid | OnlineWorkerUnavailable) {
        println(
          "coalesced-progress cancelled=\{cancelled} attempt=\{attempts} status=\{Repr(status)}",
        )
      }
      assert_true(
        status
        is (OnlineWorkerAdvanced | OnlineWorkerWaiting | OnlineWorkerRetryable),
      )
      attempts += 1
    }
    assert_true(lease.has_publication())
    // Published output remains a caller cut, not speculative successor work.
    try lease.progress() catch {
      @worker_service.WorkerServiceError::Invalid(Lifecycle) => ()
      _ => raise Failure("unexpected publication cut error")
    } noraise {
      _ => raise Failure("submitted across unconsumed publication")
    }
    if !cancelled {
      assert_eq(lease.reserve_cancel(), OnlineCancelReserved)
      assert_eq(lease.commit_cancel(), OnlineCancelCommitted)
      assert_eq(
        lease.take_publication().kind(),
        OnlineSuppressedTokenPublication,
      )
    }
    let publication = lease.take_publication()
    assert_eq(publication.kind(), OnlineTerminalPublication)
    assert_eq(publication.terminal_reason(), OnlineCancelledByCaller)
    lease.shutdown_clean()
  }
  // Consumer acknowledgement permits exactly the next flight. Exercise both
  // transitions through the reactor API rather than the diagnostic API.
  let completed = owned_online_lease(executable, clock, fixture_root~)
  completed.admit(online_request(702UL, clock.now_millis(), max_new_tokens=2))
  for position in 0..<2 {
    let mut attempts = 0
    while !completed.has_publication() && attempts < 1000000 {
      assert_true(
        completed.progress_status()
        is (OnlineWorkerAdvanced | OnlineWorkerWaiting | OnlineWorkerRetryable),
      )
      attempts += 1
    }
    assert_true(completed.has_publication())
    let token = completed.take_publication()
    assert_eq(token.kind(), OnlineGeneratedTokenPublication)
    assert_eq(token.position(), position)
  }
  assert_eq(completed.take_publication().kind(), OnlineTerminalPublication)
  completed.shutdown_clean()
}

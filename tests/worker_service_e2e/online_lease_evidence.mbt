///|
fn owned_online_service(
  executable : Bytes,
) -> @worker_service.WorkerService raise {
  let model = @approved_fs.ApprovedRoot::open_absolute("/private/tmp")
  let kernel = @approved_fs.ApprovedRoot::open_absolute("/private/tmp")
  let prepared = @worker_service.prepare_owned(
    scheduler_blueprint(),
    service_binding(),
    executable,
    startup(),
    bootstrap_source(),
    process_limits(),
    model,
    kernel,
  )
  model.close()
  kernel.close()
  prepared.take_ready()
}

///|
fn online_progress_to_terminal(
  lease : @worker_service.OnlineWorkerLease,
  expected : UInt64,
) -> Unit raise {
  match lease.progress() {
    ServiceStarted(sequence) => assert_eq(sequence, expected)
    _ => raise Failure("online service did not start exchange")
  }
  let mut attempts = 0
  while attempts < 1000000 {
    match lease.progress() {
      ServicePending(sequence) => {
        assert_eq(sequence, expected)
        attempts += 1
      }
      ServiceCompletionReady(sequence) => {
        assert_eq(sequence, expected)
        break
      }
      _ => raise Failure("unexpected online exchange progress")
    }
  }
  match lease.progress() {
    ServiceCommitted(sequence) => assert_eq(sequence, expected)
    _ => raise Failure("online service did not commit exchange")
  }
}

///|
fn retire_cut_online_flight(
  lease : @worker_service.OnlineWorkerLease,
  expected : UInt64,
) -> Unit raise {
  let mut attempts = 0
  while attempts < 1000000 {
    match lease.progress() {
      ServicePending(sequence) => {
        assert_eq(sequence, expected)
        attempts += 1
      }
      ServiceCompletionReady(sequence) => {
        assert_eq(sequence, expected)
        break
      }
      _ => raise Failure("unexpected cut-flight progress")
    }
  }
  match lease.progress() {
    ServiceCommitted(sequence) => assert_eq(sequence, expected)
    _ => raise Failure("cut flight did not retire")
  }
}

///|
fn run_inflight_cut_evidence(executable : Bytes) -> Unit raise {
  let clock = @monotonic_clock.MonotonicClock::system()

  // Cancel immediately after begin_exchange. The logical terminal is already
  // queued, but the stale physical completion must still retire.
  let started_service = owned_online_service(executable)
  let started = started_service.prepare_online_claim(clock)
  started.admit(online_request(301UL, clock.now_millis(), max_new_tokens=2))
  match started.progress() {
    ServiceStarted(1UL) => ()
    _ => raise Failure("online started-cut exchange did not begin")
  }
  assert_eq(started.reserve_cancel(), OnlineCancelReserved)
  assert_eq(started.commit_cancel(), OnlineCancelCommitted)
  retire_cut_online_flight(started, 1UL)
  let started_terminal = started.take_publication()
  assert_eq(started_terminal.kind(), OnlineTerminalPublication)
  assert_eq(started_terminal.terminal_reason(), OnlineCancelledByCaller)
  started.shutdown_clean()

  // The same retirement path remains valid after one pending syscall.
  let pending_service = owned_online_service(executable)
  let pending = pending_service.prepare_online_claim(clock)
  pending.admit(online_request(302UL, clock.now_millis(), max_new_tokens=2))
  assert_true(pending.progress() is ServiceStarted(_))
  assert_true(pending.progress() is ServicePending(_))
  assert_eq(pending.reserve_cancel(), OnlineCancelReserved)
  assert_eq(pending.commit_cancel(), OnlineCancelCommitted)
  retire_cut_online_flight(pending, 1UL)
  let pending_terminal = pending.take_publication()
  assert_eq(pending_terminal.kind(), OnlineTerminalPublication)
  assert_eq(pending_terminal.terminal_reason(), OnlineCancelledByCaller)
  pending.shutdown_clean()

  // A cut terminal cannot be dequeued while stale physical work remains. The
  // rejection is non-mutating; after exact retirement the terminal is visible,
  // and terminal-phase progress cannot build another plan.
  let terminal_first_service = owned_online_service(executable)
  let terminal_first = terminal_first_service.prepare_online_claim(clock)
  terminal_first.admit(
    online_request(304UL, clock.now_millis(), max_new_tokens=2),
  )
  assert_true(terminal_first.progress() is ServiceStarted(_))
  assert_eq(terminal_first.reserve_cancel(), OnlineCancelReserved)
  assert_eq(terminal_first.commit_cancel(), OnlineCancelCommitted)
  assert_false(terminal_first.has_publication())
  try terminal_first.take_publication() catch {
    Invalid(Lifecycle) => ()
    _ => raise Failure("unexpected live-flight terminal-dequeue error")
  } noraise {
    _ => raise Failure("cut terminal escaped before physical retirement")
  }
  retire_cut_online_flight(terminal_first, 1UL)
  assert_true(terminal_first.has_publication())
  let terminal_after_retire = terminal_first.take_publication()
  assert_eq(terminal_after_retire.kind(), OnlineTerminalPublication)
  assert_eq(terminal_after_retire.terminal_reason(), OnlineCancelledByCaller)
  try terminal_first.progress() catch {
    Invalid(Lifecycle) => ()
    _ => raise Failure("unexpected post-terminal progress error")
  } noraise {
    _ => raise Failure("terminal-phase progress built a new plan")
  }
  terminal_first.shutdown_clean()

  // Deadline mutation likewise cannot strand the already-started exchange.
  let deadline_service = owned_online_service(executable)
  let deadline = deadline_service.prepare_online_claim(clock)
  let receipt = clock.now_millis()
  deadline.admit(
    online_request(303UL, receipt, max_new_tokens=2, deadline_budget=1UL),
  )
  assert_true(deadline.progress() is ServiceStarted(_))
  let mut expiry = deadline.expire()
  while expiry == OnlineDeadlinePending {
    expiry = deadline.expire()
  }
  assert_eq(expiry, OnlineDeadlineCommitted)
  retire_cut_online_flight(deadline, 1UL)
  let deadline_terminal = deadline.take_publication()
  assert_eq(deadline_terminal.kind(), OnlineTerminalPublication)
  assert_eq(deadline_terminal.terminal_reason(), OnlineDeadlineReached)
  deadline.shutdown_clean()
}

///|
fn expect_online_alias_guard(
  action : () -> Unit raise @worker_service.WorkerServiceError,
) -> Unit raise {
  try action() catch {
    Invalid(OwnershipMode) => ()
    _ => raise Failure("unexpected online ownership guard error")
  } noraise {
    _ => raise Failure("raw alias crossed online ownership mode")
  }
}

///|
fn run_online_lease_evidence(executable : Bytes) -> Unit raise {
  let empty_service = owned_online_service(executable)
  let empty_clock = @monotonic_clock.MonotonicClock::system()
  empty_service.prepare_online_claim(empty_clock).shutdown_clean_empty()
  assert_eq(empty_service.lifecycle(), ServiceClosed)

  let service = owned_online_service(executable)
  // Eligible is neither raw nor online-active; a rejected raw alias call does
  // not consume the future online family choice.
  expect_online_alias_guard(fn() raise @worker_service.WorkerServiceError {
    ignore(service.live_request_count())
  })
  let clock = @monotonic_clock.MonotonicClock::system()
  let lease = service.prepare_online_claim(clock)
  let stale_lease = lease
  expect_online_alias_guard(fn() raise @worker_service.WorkerServiceError {
    ignore(service.outstanding_count())
  })
  expect_online_alias_guard(fn() raise @worker_service.WorkerServiceError {
    ignore(service.progress())
  })
  expect_online_alias_guard(fn() raise @worker_service.WorkerServiceError {
    ignore(service.take_next_publication())
  })
  expect_online_alias_guard(fn() raise @worker_service.WorkerServiceError {
    service.begin_recovery()
  })
  expect_online_alias_guard(fn() raise @worker_service.WorkerServiceError {
    service.close()
  })
  expect_online_alias_guard(fn() raise @worker_service.WorkerServiceError {
    service.claim_raw()
  })
  let now = clock.now_millis()
  lease.admit(online_request(201UL, now, max_new_tokens=2))
  online_progress_to_terminal(lease, 1UL)
  assert_true(lease.has_publication())
  try lease.progress() catch {
    Invalid(Lifecycle) => ()
    _ => raise Failure("unexpected queued-publication progress error")
  } noraise {
    _ => raise Failure("online progress advanced past queued publication")
  }
  assert_eq(lease.reserve_cancel(), OnlineCancelReserved)
  assert_eq(lease.commit_cancel(), OnlineCancelCommitted)
  let token = lease.take_publication()
  assert_eq(token.kind(), OnlineSuppressedTokenPublication)
  assert_eq(token.publication_sequence(), 1UL)
  let terminal = lease.take_publication()
  assert_eq(terminal.kind(), OnlineTerminalPublication)
  assert_eq(terminal.terminal_reason(), OnlineCancelledByCaller)
  assert_eq(terminal.publication_sequence(), 2UL)
  try lease.close_terminal() catch {
    Invalid(Lifecycle) => ()
    _ => raise Failure("unexpected healthy terminal-close rejection")
  } noraise {
    _ => raise Failure("healthy terminal bypassed clean shutdown")
  }
  lease.release()
  expect_online_alias_guard(fn() raise @worker_service.WorkerServiceError {
    ignore(stale_lease.has_publication())
  })
  expect_online_alias_guard(fn() raise @worker_service.WorkerServiceError {
    service.claim_raw()
  })

  // A later online lease rotates the private epoch but continues the global
  // publication sequence rather than resetting it to one.
  let second = service.prepare_online_claim(clock)
  expect_online_alias_guard(fn() raise @worker_service.WorkerServiceError {
    ignore(stale_lease.has_publication())
  })
  second.admit(online_request(202UL, clock.now_millis(), max_new_tokens=2))
  online_progress_to_terminal(second, 2UL)
  assert_eq(second.reserve_cancel(), OnlineCancelReserved)
  let second_token = second.take_reserved_token()
  assert_eq(second_token.position(), 0)
  assert_eq(second_token.publication_sequence(), 3UL)
  assert_eq(second.commit_cancel(), OnlineCancelCommitted)
  let second_terminal = second.take_publication()
  assert_eq(second_terminal.terminal_reason(), OnlineCancelledByCaller)
  assert_eq(second_terminal.publication_sequence(), 4UL)
  second.release()

  let final_lease = service.prepare_online_claim(clock)
  final_lease.admit(online_request(203UL, clock.now_millis()))
  online_progress_to_terminal(final_lease, 3UL)
  assert_eq(final_lease.expire(), OnlineDeadlineNaturalTerminal)
  assert_eq(final_lease.take_publication().publication_sequence(), 5UL)
  assert_eq(final_lease.take_publication().publication_sequence(), 6UL)
  final_lease.release()

  // Deadline cut after a token has already been queued authenticates and
  // suppresses that old-generation token before the fresh deadline terminal.
  let deadline_lease = service.prepare_online_claim(clock)
  let deadline_receipt = clock.now_millis()
  deadline_lease.admit(
    online_request(
      205UL,
      deadline_receipt,
      max_new_tokens=2,
      deadline_budget=1UL,
    ),
  )
  online_progress_to_terminal(deadline_lease, 4UL)
  let mut expiry = deadline_lease.expire()
  while expiry == OnlineDeadlinePending {
    expiry = deadline_lease.expire()
  }
  assert_eq(expiry, OnlineDeadlineCommitted)
  let deadline_token = deadline_lease.take_publication()
  assert_eq(deadline_token.kind(), OnlineSuppressedTokenPublication)
  assert_eq(deadline_token.publication_sequence(), 7UL)
  let deadline_terminal = deadline_lease.take_publication()
  assert_eq(deadline_terminal.terminal_reason(), OnlineDeadlineReached)
  assert_eq(deadline_terminal.publication_sequence(), 8UL)
  deadline_lease.shutdown_clean()
  assert_eq(service.lifecycle(), ServiceClosed)

  // Online recovery retains the exact request generation, invalidates its
  // device state before replacement, and authenticates the fresh failure
  // terminal without exposing a raw scheduler handle.
  let recovery_service = owned_online_service(executable)
  let recovery_lease = recovery_service.prepare_online_claim(clock)
  recovery_lease.admit(
    online_request(204UL, clock.now_millis(), max_new_tokens=2),
  )
  online_progress_to_terminal(recovery_lease, 1UL)
  assert_eq(
    recovery_lease.take_publication().kind(),
    OnlineGeneratedTokenPublication,
  )
  recovery_lease.begin_recovery()
  assert_eq(recovery_service.lifecycle(), ServiceDeviceInvalidationRequired)
  assert_eq(recovery_lease.invalidate_device_state(), 1)
  assert_eq(recovery_service.lifecycle(), ServiceRestartReady)
  recovery_lease.abandon_restart()
  assert_eq(recovery_lease.drain_restart_forbidden(), Complete)
  assert_eq(recovery_service.lifecycle(), ServiceTerminalCloseReady)
  let failed = recovery_lease.take_publication()
  assert_eq(failed.kind(), OnlineTerminalPublication)
  assert_eq(failed.terminal_reason(), OnlineWorkerFailed)
  recovery_lease.close_terminal()
  assert_eq(recovery_service.lifecycle(), ServiceClosed)

  run_inflight_cut_evidence(executable)
}

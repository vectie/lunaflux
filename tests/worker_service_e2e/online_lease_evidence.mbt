///|
fn owned_online_lease(
  executable : Bytes,
  clock : @monotonic_clock.MonotonicClock,
) -> @worker_service.OnlineWorkerLease raise {
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
  prepared.take_online(clock)
}

///|
fn run_owned_online_transfer_evidence(executable : Bytes) -> Unit raise {
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
  let lease = prepared.take_online(@monotonic_clock.MonotonicClock::system())
  try prepared.take_online(@monotonic_clock.MonotonicClock::system()) catch {
    OwnedPreparationFailure(OwnedServiceFailure(Invalid(Lifecycle))) => ()
    _ => raise Failure("unexpected repeated online extraction error")
  } noraise {
    _ => raise Failure("online preparation extracted twice")
  }
  try prepared.take_raw_ready() catch {
    OwnedPreparationFailure(OwnedServiceFailure(Invalid(Lifecycle))) => ()
    _ => raise Failure("unexpected online-to-raw extraction error")
  } noraise {
    _ => raise Failure("online preparation also exposed raw service")
  }
  try prepared.take_cleanup() catch {
    OwnedPreparationFailure(OwnedServiceFailure(Invalid(Lifecycle))) => ()
    _ => raise Failure("unexpected online cleanup extraction error")
  } noraise {
    _ => raise Failure("online preparation also exposed cleanup")
  }
  lease.shutdown_clean_empty()
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
  let started = owned_online_lease(executable, clock)
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
  let pending = owned_online_lease(executable, clock)
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
  let terminal_first = owned_online_lease(executable, clock)
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
  let deadline = owned_online_lease(executable, clock)
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
fn run_online_lease_evidence(executable : Bytes) -> Unit raise {
  let clock = @monotonic_clock.MonotonicClock::system()

  run_owned_online_transfer_evidence(executable)

  let empty = owned_online_lease(executable, clock)
  empty.shutdown_clean_empty()

  let cancelled = owned_online_lease(executable, clock)
  cancelled.admit(online_request(201UL, clock.now_millis(), max_new_tokens=2))
  online_progress_to_terminal(cancelled, 1UL)
  assert_eq(cancelled.reserve_cancel(), OnlineCancelReserved)
  assert_eq(cancelled.commit_cancel(), OnlineCancelCommitted)
  assert_eq(
    cancelled.take_publication().kind(),
    OnlineSuppressedTokenPublication,
  )
  let terminal = cancelled.take_publication()
  assert_eq(terminal.kind(), OnlineTerminalPublication)
  assert_eq(terminal.terminal_reason(), OnlineCancelledByCaller)
  cancelled.shutdown_clean()

  let reserved = owned_online_lease(executable, clock)
  reserved.admit(online_request(202UL, clock.now_millis(), max_new_tokens=2))
  online_progress_to_terminal(reserved, 1UL)
  assert_eq(reserved.reserve_cancel(), OnlineCancelReserved)
  assert_eq(reserved.take_reserved_token().position(), 0)
  assert_eq(reserved.commit_cancel(), OnlineCancelCommitted)
  assert_eq(
    reserved.take_publication().terminal_reason(),
    OnlineCancelledByCaller,
  )
  reserved.shutdown_clean()

  let deadline = owned_online_lease(executable, clock)
  let receipt = clock.now_millis()
  deadline.admit(
    online_request(205UL, receipt, max_new_tokens=2, deadline_budget=1UL),
  )
  online_progress_to_terminal(deadline, 1UL)
  let mut expiry = deadline.expire()
  while expiry == OnlineDeadlinePending {
    expiry = deadline.expire()
  }
  assert_eq(expiry, OnlineDeadlineCommitted)
  assert_eq(
    deadline.take_publication().kind(),
    OnlineSuppressedTokenPublication,
  )
  assert_eq(
    deadline.take_publication().terminal_reason(),
    OnlineDeadlineReached,
  )
  deadline.shutdown_clean()

  let recovery = owned_online_lease(executable, clock)
  recovery.admit(online_request(204UL, clock.now_millis(), max_new_tokens=2))
  online_progress_to_terminal(recovery, 1UL)
  assert_eq(recovery.take_publication().kind(), OnlineGeneratedTokenPublication)
  recovery.begin_recovery()
  assert_eq(recovery.invalidate_device_state(), 1)
  recovery.abandon_restart()
  assert_eq(recovery.drain_restart_forbidden(), Complete)
  let failed = recovery.take_publication()
  assert_eq(failed.kind(), OnlineTerminalPublication)
  assert_eq(failed.terminal_reason(), OnlineWorkerFailed)
  recovery.close_terminal()

  run_inflight_cut_evidence(executable)
}

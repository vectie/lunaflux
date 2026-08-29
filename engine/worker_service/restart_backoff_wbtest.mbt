///|
test "restart backoff policy rejects unbounded geometry" {
  for
    policy in [
      (0UL, 8UL, 4UL, 8),
      (9UL, 8UL, 4UL, 8),
      (1UL, 0x80000000UL, 4UL, 8),
      (1UL, 8UL, 0UL, 8),
      (1UL, 8UL, 4UL, 0),
    ] {
    try
      WorkerRestartBackoffPolicy::new(
        initial_delay_millis=policy.0,
        maximum_delay_millis=policy.1,
        stable_success_millis=policy.2,
        maximum_attempts=policy.3,
      )
    catch {
      Invalid(RestartBackoff) => ()
      _ => fail("unexpected restart policy error")
    } noraise {
      _ => fail("invalid restart policy was accepted")
    }
  }
}

///|
test "restart delay doubles saturates and exhausts without wrapping" {
  let lease = wb_empty_online_lease(@monotonic_clock.MonotonicClock::system())
  let expected : FixedArray[UInt64] = [1UL, 2UL, 4UL, 8UL, 8UL, 8UL, 8UL, 8UL]
  for index in 0..<expected.length() {
    assert_true(lease.schedule_restart_backoff_at(index.to_uint64()))
    assert_eq(lease.restart_attempts, index + 1)
    assert_eq(lease.restart_generation, (index + 1).to_uint64())
    assert_eq(lease.restart_delay_millis, expected[index])
    lease.restart_phase = 0
  }
  assert_false(lease.schedule_restart_backoff_at(8UL))
  assert_true(lease.restart_forbidden)
  assert_eq(lease.restart_attempts, 8)
  assert_eq(lease.restart_generation, 8UL)
}

///|
test "restart timestamp overflow and rollback fail closed" {
  let overflow = wb_empty_online_lease(
    @monotonic_clock.MonotonicClock::system(),
  )
  assert_false(overflow.schedule_restart_backoff_at(0xffffffffffffffffUL))
  assert_true(overflow.restart_forbidden)
  assert_eq(overflow.restart_attempts, 0)
  assert_eq(overflow.restart_generation, 0UL)

  let rollback = wb_empty_online_lease(
    @monotonic_clock.MonotonicClock::system(),
  )
  assert_true(rollback.schedule_restart_backoff_at(10UL))
  assert_eq(rollback.restart_backoff_remaining_at(9UL), 0)
  assert_true(rollback.restart_forbidden)
  assert_eq(rollback.restart_phase, 4)
}

///|
test "replacement stability deadline overflow is sticky forbidden" {
  let lease = wb_empty_online_lease(@monotonic_clock.MonotonicClock::system())
  assert_true(lease.schedule_restart_backoff_at(1UL))
  lease.restart_phase = 2
  lease.record_replacement_ready_at(0xffffffffffffffffUL, 3UL)
  assert_true(lease.restart_forbidden)
  assert_eq(lease.restart_phase, 4)
  assert_eq(lease.restart_stability_generation, 0UL)
  assert_eq(lease.restart_stability_deadline_millis, 0UL)
}

///|
test "restart generation exhaustion fails closed before attempt mutation" {
  let lease = wb_empty_online_lease(@monotonic_clock.MonotonicClock::system())
  lease.restart_generation = 0xffffffffffffffffUL
  assert_false(lease.schedule_restart_backoff_at(1UL))
  assert_true(lease.restart_forbidden)
  assert_eq(lease.restart_attempts, 0)
  assert_eq(lease.restart_generation, 0xffffffffffffffffUL)
}

///|
test "restart wake query is observational and returns cached bound" {
  let lease = wb_empty_online_lease(@monotonic_clock.MonotonicClock::system())
  lease.service.lifecycle = ServiceRestartReady
  assert_true(lease.schedule_restart_backoff_at(50UL))
  let phase = lease.restart_phase
  let last_clock = lease.restart_last_clock_millis
  let deadline = lease.restart_deadline_millis
  assert_eq(try! lease.restart_backoff_remaining_millis(), 1)
  assert_eq(lease.restart_phase, phase)
  assert_eq(lease.restart_last_clock_millis, last_clock)
  assert_eq(lease.restart_deadline_millis, deadline)
  assert_false(lease.restart_forbidden)
}

///|
test "stable reset requires current replacement generation and deadline" {
  let lease = wb_empty_online_lease(@monotonic_clock.MonotonicClock::system())
  assert_true(lease.schedule_restart_backoff_at(100UL))
  lease.restart_phase = 2
  lease.record_replacement_ready_at(101UL, 40UL)
  assert_eq(lease.restart_stability_generation, 1UL)
  assert_eq(lease.restart_stability_deadline_millis, 105UL)

  lease.record_stable_replacement_commit_at(104UL, 41UL, 41UL, true)
  assert_eq(lease.restart_attempts, 1)
  lease.restart_stability_generation = 2UL
  lease.record_stable_replacement_commit_at(105UL, 41UL, 41UL, true)
  assert_eq(lease.restart_attempts, 1)

  lease.restart_stability_generation = lease.restart_generation
  lease.record_stable_replacement_commit_at(105UL, 40UL, 40UL, true)
  assert_eq(lease.restart_attempts, 1)
  lease.record_stable_replacement_commit_at(105UL, 41UL, 40UL, true)
  assert_eq(lease.restart_attempts, 1)
  lease.record_stable_replacement_commit_at(105UL, 41UL, 41UL, false)
  assert_eq(lease.restart_attempts, 1)
  lease.record_stable_replacement_commit_at(105UL, 41UL, 41UL, true)
  assert_eq(lease.restart_attempts, 0)
  assert_eq(lease.restart_phase, 0)
  assert_eq(lease.restart_generation, 1UL)
}

///|
test "replacement cleanup completes before a new delay can begin" {
  let lease = wb_empty_online_lease(@monotonic_clock.MonotonicClock::system())
  let service = lease.service
  service.lifecycle = ServiceReplacementCleanup
  service.maintenance_mode = MAINTENANCE_REPLACEMENT_CLEANUP
  service.maintenance_phase = MAINTENANCE_CHILD
  assert_eq(
    try! service.apply_child_maintenance_progress(
      false, false, false, false, true,
    ),
    LunaWorkerServiceRestartReady,
  )
  assert_eq(service.lifecycle, ServiceRestartReady)
  assert_eq(lease.restart_attempts, 0)
  assert_eq(lease.restart_phase, 0)
  assert_true(lease.schedule_restart_backoff_at(20UL))
}

///|
test "replacement cleanup failure retains exact owner and cannot begin delay" {
  let lease = wb_empty_online_lease(@monotonic_clock.MonotonicClock::system())
  let service = lease.service
  service.lifecycle = ServiceReplacementCleanup
  service.maintenance_mode = MAINTENANCE_REPLACEMENT_CLEANUP
  service.maintenance_phase = MAINTENANCE_CHILD
  assert_eq(
    try! service.apply_child_maintenance_progress(
      false, true, false, false, false,
    ),
    LunaWorkerServiceMaintenanceCleanupStuck,
  )
  assert_eq(service.lifecycle, ServiceReplacementCleanup)
  assert_eq(service.maintenance_mode, MAINTENANCE_REPLACEMENT_CLEANUP)
  assert_eq(service.maintenance_phase, MAINTENANCE_CHILD)
  assert_eq(lease.restart_attempts, 0)
  assert_eq(lease.restart_phase, 0)
  try lease.progress_restart_backoff() catch {
    Invalid(Lifecycle) => ()
    _ => fail("unexpected replacement cleanup error")
  } noraise {
    _ => fail("replacement cleanup was bypassed by restart delay")
  }
}

///|
test "drain during restart wait reuses deterministic fail-close maintenance" {
  let lease = wb_empty_online_lease(@monotonic_clock.MonotonicClock::system())
  let service = lease.service
  service.lifecycle = ServiceRestartReady
  assert_true(lease.schedule_restart_backoff_at(30UL))
  try! service.abandon_restart_impl()
  try! service.begin_recovery_maintenance_impl(ResourceUnavailable)
  assert_eq(service.lifecycle, ServiceRestartForbidden)
  assert_eq(service.maintenance_mode, MAINTENANCE_RECOVERY)
  assert_eq(service.maintenance_phase, MAINTENANCE_DRAIN)
  assert_eq(lease.restart_attempts, 1)
  assert_eq(lease.restart_phase, 1)
}

///|
test "caller cancellation terminal survives restart-wait abandon and drain" {
  let scheduler = wb_service_scheduler()
  let handle = wb_admit_routed(scheduler, wb_online_request())
  let lease = wb_restarted_lease(
    scheduler,
    handle,
    ONLINE_LANE_LIVE,
    handle.request_generation(),
    0,
  )
  assert_eq(try! lease.reserve_cancel(), OnlineCancelReserved)
  assert_eq(try! lease.commit_cancel(), OnlineCancelCommitted)
  lease.service.lifecycle = ServiceRestartReady
  assert_true(lease.schedule_restart_backoff_at(40UL))

  try! lease.abandon_restart()
  try! lease.begin_recovery_maintenance(ResourceUnavailable)
  assert_eq(try! lease.progress_maintenance(), LunaWorkerServiceFailCloseReady)
  let publication = try! lease.take_publication()
  assert_eq(publication.kind(), OnlineTerminalPublication)
  assert_eq(publication.terminal_reason(), OnlineCancelledByCaller)
  assert_eq(lease.restart_attempts, 1)
  assert_eq(lease.restart_phase, 1)
}

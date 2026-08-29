///|
test "Luna disconnect resumes retirement after abort seam" {
  let compatibility = coordinator_fixture()
  let service = compatibility.service
  // Model the failure-recovery seam after the lower owner has committed to
  // retirement: the first abort attempt may reject because the instance has
  // already crossed its abortable phase, but maintenance_kind retains the
  // exact retirement obligation and restart observation.
  service.tickets.push({ lane: -1, epoch: 1UL })
  service.ticket_sequences.push(1UL)
  service.instance.restart_observation_pending = true
  try service.begin_output_disconnect() catch {
    LunaOnlineFramedCoordinatorFailed(LunaOnlineFramedMaintenance) => ()
    _ => fail("unexpected initial disconnect seam error")
  } noraise {
    _ => fail("fixture did not reproduce initial abort seam")
  }
  assert_eq(service.maintenance_kind, 2)
  assert_true(service.disconnect_requested)
  assert_true(service.instance.restart_observation_pending)

  // Regression: reactor and maintenance re-entry must not invoke begin_abort
  // again. It leaves the ticket and observation pinned for the existing
  // progress_request_retirement branch.
  service.begin_output_disconnect()
  assert_eq(service.maintenance_kind, 2)
  assert_true(service.instance.restart_observation_pending)
  try service.progress_maintenance_core() catch {
    LunaOnlineFramedCoordinatorFailed(LunaOnlineFramedMaintenance) => ()
    _ => fail("unexpected retirement-path fixture error")
  } noraise {
    _ => fail("maintenance failed to enter request retirement")
  }
}

///|
test "post-restart FailureUsage cut retains healthy lease release path" {
  let instance = coordinator_fixture_instance()
  instance.phase = LunaRequestActivePhase
  instance.event_kind = FailureUsageEvent
  instance.failure_kind = WorkerLatchedFailure
  instance.recovery_close = false
  instance.restarted_failure_terminal_pending = false

  assert_true(instance.finish_abort_after_event_cut())
  assert_true(instance.phase == LunaRequestReleaseReadyPhase)
  assert_false(instance.recovery_close)
  assert_true(instance.failure_kind == WorkerLatchedFailure)

  // The same public failure event on a truly unrecoverable owner must retain
  // close-only cleanup rather than publish a reusable instance.
  instance.phase = LunaRequestActivePhase
  instance.recovery_close = true
  assert_true(instance.finish_abort_after_event_cut())
  assert_true(instance.phase == LunaInstanceClosingTerminalPhase)
}

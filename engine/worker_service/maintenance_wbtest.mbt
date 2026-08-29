///|
fn wb_maintenance_service(
  scheduler : @core.Scheduler,
  flight? : WorkerFlightState = FlightEmpty,
  pending_plan_sequence? : UInt64 = 0UL,
  lifecycle? : WorkerServiceLifecycle = ServiceRecovering,
  maintenance_mode? : Int = MAINTENANCE_RECOVERY,
  maintenance_phase? : Int = MAINTENANCE_RETIRE,
) -> WorkerService {
  {
    scheduler,
    identity: wb_worker_service_identity(),
    physical_binding: RootedSingleBinding(wb_service_binding()),
    transport: Uninstalled,
    flight,
    pending_plan_sequence,
    maintenance_mode,
    maintenance_phase,
    maintenance_failure: ExecutionFailed,
    lifecycle,
    production_owned: false,
    ownership_mode: FixtureRawOwnership,
    online_epoch: 0UL,
    online_publication_cursor: 0UL,
    graph_telemetry: new_worker_service_graph_telemetry(),
  }
}

///|
fn wb_submitted_scheduler(
  request_value : UInt64,
) -> (@core.Scheduler, @core.RequestHandle, UInt64) {
  let scheduler = wb_service_scheduler()
  let request = wb_retirement_request(request_value, 0UL)
  let prepared = scheduler.prepare_admission(request)
  try! scheduler.commit_admission(prepared, 0UL)
  let handle = try! prepared.request_handle()
  let outcome = try! scheduler.build_next()
  let plan = try! scheduler.submitted_plan(outcome)
  (scheduler, handle, try! plan.sequence_wire())
}

///|
test "cooperative child disposition is scalar and never spins phases" {
  let service = wb_maintenance_service(
    wb_service_scheduler(),
    maintenance_phase=MAINTENANCE_CHILD,
  )
  assert_eq(
    try! service.apply_child_maintenance_progress(
      false, false, true, false, false,
    ),
    LunaWorkerServiceMaintenancePending,
  )
  assert_eq(service.maintenance_phase, MAINTENANCE_CHILD)
  assert_eq(
    try! service.apply_child_maintenance_progress(
      false, false, false, true, false,
    ),
    LunaWorkerServiceMaintenanceAdvanced,
  )
  assert_eq(
    try! service.apply_child_maintenance_progress(
      true, false, false, false, false,
    ),
    LunaWorkerServiceMaintenanceCleanupRequired,
  )
  assert_eq(
    try! service.apply_child_maintenance_progress(
      false, true, false, false, false,
    ),
    LunaWorkerServiceMaintenanceCleanupStuck,
  )
  assert_eq(service.maintenance_phase, MAINTENANCE_CHILD)
  assert_eq(service.lifecycle, ServiceRecovering)
  assert_eq(
    try! service.apply_child_maintenance_progress(
      false, false, false, false, true,
    ),
    LunaWorkerServiceRecoveryReady,
  )
  assert_eq(service.maintenance_phase, MAINTENANCE_RETIRE)
  assert_eq(try! service.maintenance_remaining_millis_impl(), 0)
}

///|
test "empty recovery and clean shutdown each finish one phase per call" {
  let recovery = wb_maintenance_service(
    wb_service_scheduler(),
    maintenance_phase=MAINTENANCE_CHILD,
  )
  assert_eq(
    try! recovery.apply_child_maintenance_progress(
      false, false, false, false, true,
    ),
    LunaWorkerServiceRecoveryReady,
  )
  assert_eq(
    try! recovery.progress_maintenance_impl(),
    LunaWorkerServiceMaintenanceAdvanced,
  )
  assert_eq(recovery.maintenance_phase, MAINTENANCE_INVALIDATE)
  assert_eq(
    try! recovery.progress_maintenance_impl(),
    LunaWorkerServiceRestartReady,
  )

  let shutdown = wb_maintenance_service(
    wb_service_scheduler(),
    lifecycle=ServiceClosingFailed,
    maintenance_mode=MAINTENANCE_SHUTDOWN,
    maintenance_phase=MAINTENANCE_CHILD,
  )
  assert_eq(
    try! shutdown.apply_child_maintenance_progress(
      false, false, false, false, true,
    ),
    LunaWorkerServiceClosed,
  )
  assert_eq(shutdown.lifecycle, ServiceClosed)
  assert_eq(shutdown.maintenance_mode, MAINTENANCE_NONE)
}

///|
test "replacement cleanup completion restores restart-ready cooperatively" {
  let service = wb_maintenance_service(
    wb_service_scheduler(),
    lifecycle=ServiceReplacementCleanup,
    maintenance_mode=MAINTENANCE_REPLACEMENT_CLEANUP,
    maintenance_phase=MAINTENANCE_CHILD,
  )
  assert_eq(
    try! service.apply_child_maintenance_progress(
      false, true, false, false, false,
    ),
    LunaWorkerServiceMaintenanceCleanupStuck,
  )
  assert_eq(service.lifecycle, ServiceReplacementCleanup)
  assert_eq(service.maintenance_mode, MAINTENANCE_REPLACEMENT_CLEANUP)
  assert_eq(service.maintenance_phase, MAINTENANCE_CHILD)
  assert_eq(
    try! service.apply_child_maintenance_progress(
      false, false, false, false, true,
    ),
    LunaWorkerServiceRestartReady,
  )
  assert_eq(service.lifecycle, ServiceRestartReady)
  assert_eq(service.maintenance_mode, MAINTENANCE_NONE)
  assert_eq(service.maintenance_phase, 0)
}

///|
test "physical and pending recovery obligations remain separate" {
  let service = wb_maintenance_service(
    wb_service_scheduler(),
    flight=FlightSubmitted(31UL),
    pending_plan_sequence=32UL,
  )
  assert_eq(service.finish_recovered_flight(31UL, false), 31UL)
  assert_true(service.flight is FlightPendingFailure(32UL, 31UL, false))
  assert_eq(service.pending_plan_sequence, 0UL)
  assert_eq(service.lifecycle, ServiceRecovering)
}

///|
test "pending scheduler side retires on its own maintenance turn" {
  let (scheduler, _, sequence) = wb_submitted_scheduler(1201UL)
  let service = wb_maintenance_service(
    scheduler,
    flight=FlightPendingFailure(sequence, sequence, false),
  )
  assert_eq(
    try! service.progress_maintenance_impl(),
    LunaWorkerServiceMaintenanceAdvanced,
  )
  assert_true(service.flight is FlightEmpty)
  assert_eq(service.maintenance_phase, MAINTENANCE_INVALIDATE)
  assert_eq(service.lifecycle, ServiceDeviceInvalidationRequired)
  assert_eq(
    try! service.progress_maintenance_impl(),
    LunaWorkerServiceRestartReady,
  )
  assert_eq(service.maintenance_mode, MAINTENANCE_NONE)
  assert_eq(try! scheduler.take_terminal_notice().reason(), WorkerFailed)
}

///|
test "completion publication backpressure preserves pending plan authority" {
  let scheduler = wb_service_scheduler()
  let target = wb_retirement_request(1211UL, 0UL)
  let target_prepared = scheduler.prepare_admission(target)
  try! scheduler.commit_admission(target_prepared, 0UL)
  let outcome = try! scheduler.build_next()
  let sequence = try! (try! scheduler.submitted_plan(outcome)).sequence_wire()
  for offset in 0..<6 {
    let cancelled = scheduler.prepare_admission(
      wb_retirement_request(1240UL + offset.to_uint64(), 0UL),
    )
    try! scheduler.commit_admission(cancelled, 0UL)
    ignore(try! scheduler.cancel_exact(try! cancelled.request_handle()))
  }
  assert_eq(scheduler.terminal_notice_count(), 6)
  let service = wb_maintenance_service(
    scheduler,
    flight=FlightPendingFailure(sequence, sequence, false),
  )
  assert_eq(
    try! service.progress_maintenance_impl(),
    LunaWorkerServiceMaintenancePending,
  )
  assert_true(service.flight is FlightPendingFailure(_, _, false))
  assert_eq(service.flight_sequence_value(), sequence)
  assert_eq(service.maintenance_phase, MAINTENANCE_RETIRE)
  ignore(try! scheduler.take_terminal_notice())
  assert_eq(
    try! service.progress_maintenance_impl(),
    LunaWorkerServiceMaintenanceAdvanced,
  )
  assert_true(service.flight is FlightEmpty)
}

///|
fn wb_cut_pending_plan(deadline : Bool) -> Unit raise {
  let (scheduler, handle, sequence) = wb_submitted_scheduler(
    if deadline {
      1222UL
    } else {
      1221UL
    },
  )
  if deadline {
    assert_eq(scheduler.expire_deadlines(1000UL), 1)
  } else {
    ignore(try! scheduler.cancel_exact(handle))
  }
  let service = wb_maintenance_service(
    scheduler,
    flight=FlightPendingFailure(sequence, sequence, false),
  )
  assert_eq(
    try! service.progress_maintenance_impl(),
    LunaWorkerServiceMaintenanceAdvanced,
  )
  assert_true(service.flight is FlightEmpty)
  assert_eq(service.maintenance_phase, MAINTENANCE_INVALIDATE)
}

///|
test "cancelled and expired pending sides retire without substitution" {
  wb_cut_pending_plan(false)
  wb_cut_pending_plan(true)
}

///|
test "stale maintenance and blocking aliases reject before lower access" {
  let idle = wb_maintenance_service(
    wb_service_scheduler(),
    lifecycle=ServiceReady,
    maintenance_mode=MAINTENANCE_NONE,
    maintenance_phase=0,
  )
  try idle.progress_maintenance() catch {
    Invalid(Lifecycle) => ()
    _ => fail("unexpected idle maintenance progress error")
  } noraise {
    _ => fail("idle owner accepted stale maintenance progress")
  }

  let active = wb_maintenance_service(wb_service_scheduler())
  try active.begin_recovery() catch {
    Invalid(Lifecycle) => ()
    _ => fail("unexpected blocking alias error")
  } noraise {
    _ => fail("blocking recovery substituted for cooperative authority")
  }
  try active.begin_recovery_maintenance(ExecutionFailed) catch {
    Invalid(Lifecycle) => ()
    _ => fail("unexpected duplicate maintenance error")
  } noraise {
    _ => fail("duplicate cooperative maintenance was accepted")
  }
}

///|
test "stale scalar pending plan is transactional during maintenance" {
  let service = wb_maintenance_service(
    wb_service_scheduler(),
    pending_plan_sequence=77UL,
  )
  try service.progress_maintenance_impl() catch {
    Scheduler(Invalid(field=Plan, issue=Stale)) => ()
    _ => fail("unexpected stale maintenance plan error")
  } noraise {
    _ => fail("stale maintenance plan reached retirement")
  }
  assert_true(service.flight is FlightEmpty)
  assert_eq(service.pending_plan_sequence, 77UL)
  assert_eq(service.maintenance_phase, MAINTENANCE_RETIRE)
  assert_eq(service.lifecycle, ServiceRecovering)
}

///|
test "clean shutdown preflights scheduler authority before child access" {
  let scheduler = wb_service_scheduler()
  let prepared = scheduler.prepare_admission(wb_retirement_request(1231UL, 0UL))
  try! scheduler.commit_admission(prepared, 0UL)
  let service = wb_maintenance_service(
    scheduler,
    lifecycle=ServiceReady,
    maintenance_mode=MAINTENANCE_NONE,
    maintenance_phase=0,
  )
  try service.begin_shutdown_maintenance() catch {
    Invalid(Lifecycle) => ()
    _ => fail("unexpected live shutdown preflight error")
  } noraise {
    _ => fail("live request reached child shutdown")
  }
  assert_eq(service.maintenance_mode, MAINTENANCE_NONE)
  assert_eq(service.lifecycle, ServiceReady)
}

///|
test "online maintenance wrappers authenticate before lower access" {
  let lease = wb_empty_online_lease(@monotonic_clock.MonotonicClock::system())
  try lease.progress_maintenance() catch {
    Invalid(Lifecycle) => ()
    _ => fail("unexpected online stale maintenance error")
  } noraise {
    _ => fail("online stale maintenance progress reached the process")
  }
  try lease.maintenance_remaining_millis() catch {
    Invalid(Lifecycle) => ()
    _ => fail("unexpected online remaining-time error")
  } noraise {
    _ => fail("online stale maintenance time query reached the process")
  }
}

///|
test "online shutdown binds request phase to semantic retention" {
  let terminal = wb_terminal_online_lease()
  try! terminal.lane_semantic_retentions[0].release()
  try terminal.begin_shutdown_maintenance() catch {
    Invalid(Lifecycle) => ()
    _ => fail("unexpected terminal retention preflight error")
  } noraise {
    _ => fail("terminal shutdown lost required semantic retention")
  }

  let empty = wb_empty_online_lease(@monotonic_clock.MonotonicClock::system())
  let (semantic_lease, stop_tokens) = wb_service_semantic_stop_authority(
    wb_service_inference_limits(),
    [],
  )
  try! stop_tokens.retain_into(empty.lane_semantic_retentions[0])
  try empty.begin_shutdown_maintenance() catch {
    Invalid(Lifecycle) => ()
    _ => fail("unexpected empty retention preflight error")
  } noraise {
    _ => fail("empty shutdown accepted foreign semantic retention")
  }
  try! empty.lane_semantic_retentions[0].release()
  try! semantic_lease.release()
}

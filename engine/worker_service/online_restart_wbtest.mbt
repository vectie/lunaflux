///|
fn wb_restarted_lease(
  scheduler : @core.Scheduler,
  handle : @core.RequestHandle,
  phase : Int,
  generation : @inference.RequestGeneration,
  disposition : Int,
) -> OnlineWorkerLease {
  let service : WorkerService = {
    scheduler,
    identity: wb_worker_service_identity(),
    physical_binding: RootedSingleBinding(wb_service_binding()),
    transport: Uninstalled,
    flight: FlightEmpty,
    pending_plan_sequence: 0UL,
    maintenance_mode: MAINTENANCE_NONE,
    maintenance_phase: 0,
    maintenance_failure: ExecutionFailed,
    lifecycle: ServiceReady,
    production_owned: true,
    ownership_mode: OnlineActiveOwnership,
    online_epoch: 1UL,
    online_publication_cursor: 0UL,
    graph_telemetry: new_worker_service_graph_telemetry(),
  }
  wb_online_lane_lease(
    service,
    @monotonic_clock.MonotonicClock::system(),
    None,
    @monotonic_clock.MonotonicClock::system().prepare_read(),
    Some(handle),
    phase,
    generation,
    @inference.RequestGeneration::initial(),
    disposition,
    0,
    false,
    @inference.LunaRequestStopTokenRetentionSlot::new(),
  )
}

///|
test "restarted waiting request becomes exact WorkerFailed terminal" {
  let scheduler = wb_service_scheduler()
  let handle = wb_admit_routed(scheduler, wb_online_request())
  let original = handle.request_generation()
  let lease = wb_restarted_lease(
    scheduler,
    handle,
    ONLINE_LANE_TERMINATION_PENDING,
    original,
    1,
  )

  lease.prepare_restarted_failure_terminal()
  assert_eq(lease.lane_phases[0], ONLINE_LANE_TERMINATION_PENDING)
  assert_eq(lease.lane_restart_dispositions[0], 1)
  assert_eq(scheduler.live_request_count(), 0)
  assert_eq(scheduler.terminal_notice_count(), 1)
  let publication = lease.take_publication()
  assert_eq(publication.kind(), OnlineTerminalPublication)
  assert_eq(publication.terminal_reason(), OnlineWorkerFailed)
  assert_eq(lease.lane_phases[0], ONLINE_LANE_TERMINAL)
  assert_eq(lease.lane_restart_dispositions[0], 0)
}

///|
test "restarted terminal-pending request is not terminalized twice" {
  let scheduler = wb_service_scheduler()
  let handle = wb_admit_routed(scheduler, wb_online_request())
  let original = handle.request_generation()
  let drained = scheduler.drain_instance_loss_exact(handle)
  let terminal_generation = match drained {
    ExactLossComplete(value) => value
    ExactLossRemaining(_, _) => abort("restarted pending fixture invariant")
  }
  scheduler.advance_terminated_handle(handle, terminal_generation)
  let lease = wb_restarted_lease(
    scheduler,
    handle,
    ONLINE_LANE_TERMINATION_PENDING,
    terminal_generation,
    2,
  )
  lease.lane_cut_source_generations[0] = original

  lease.prepare_restarted_failure_terminal()
  assert_eq(lease.lane_restart_dispositions[0], 2)
  assert_eq(scheduler.live_request_count(), 0)
  assert_eq(scheduler.terminal_notice_count(), 1)
  let publication = lease.take_publication()
  assert_eq(publication.terminal_reason(), OnlineWorkerFailed)
  assert_eq(publication.publication_sequence(), 1UL)
  assert_eq(lease.lane_restart_dispositions[0], 0)
  assert_eq(lease.lane_cut_source_generations[0], original)
}

///|
test "fail-close reuses an already queued exact WorkerFailed terminal" {
  let scheduler = wb_service_scheduler()
  let handle = wb_admit_routed(scheduler, wb_online_request())
  let original = handle.request_generation()
  ignore(try! scheduler.drain_instance_loss_exact(handle))
  let lease = wb_restarted_lease(
    scheduler,
    handle,
    ONLINE_LANE_TERMINATION_PENDING,
    original,
    2,
  )
  lease.service.lifecycle = ServiceRestartForbidden

  assert_eq(try! lease.drain_restart_forbidden(), Complete)
  assert_eq(lease.service.lifecycle, ServiceTerminalCloseReady)
  let publication = try! lease.take_publication()
  assert_eq(publication.terminal_reason(), OnlineWorkerFailed)
  assert_eq(lease.lane_phases[0], ONLINE_LANE_TERMINAL)
}

///|
test "fail-close exact drain terminalizes a retained waiting request" {
  let scheduler = wb_service_scheduler()
  let handle = wb_admit_routed(scheduler, wb_online_request())
  let original = handle.request_generation()
  let lease = wb_restarted_lease(
    scheduler,
    handle,
    ONLINE_LANE_TERMINATION_PENDING,
    original,
    1,
  )
  lease.service.lifecycle = ServiceRestartForbidden

  assert_eq(try! lease.drain_restart_forbidden(), Complete)
  assert_eq(lease.service.lifecycle, ServiceTerminalCloseReady)
  let publication = try! lease.take_publication()
  assert_eq(publication.terminal_reason(), OnlineWorkerFailed)
  assert_eq(lease.lane_phases[0], ONLINE_LANE_TERMINAL)
}

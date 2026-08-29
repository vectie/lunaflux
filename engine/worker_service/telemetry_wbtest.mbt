///|
test "online telemetry authenticates the exact lease epoch" {
  let scheduler = wb_service_scheduler()
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
    online_epoch: 7UL,
    online_publication_cursor: 0UL,
    graph_telemetry: new_worker_service_graph_telemetry(),
  }
  let lease = wb_online_lane_lease(
    service,
    @monotonic_clock.MonotonicClock::system(),
    None,
    @monotonic_clock.MonotonicClock::system().prepare_read(),
    None,
    ONLINE_LANE_EMPTY,
    @inference.RequestGeneration::initial(),
    @inference.RequestGeneration::initial(),
    0,
    0,
    false,
    @inference.LunaRequestStopTokenRetentionSlot::new(),
    epoch=7UL,
  )
  assert_eq((try! lease.luna_scheduler_telemetry()).queue_depth(), 0)
  assert_eq(
    (try! lease.luna_scheduler_plan_telemetry()).last_plan_sequence(),
    0UL,
  )
  service.online_epoch = 8UL
  try lease.luna_scheduler_telemetry() catch {
    Invalid(OwnershipMode) => ()
    _ => fail("unexpected stale telemetry error")
  } noraise {
    _ => fail("stale online lease exposed telemetry")
  }
}

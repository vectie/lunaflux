///|
test "request eviction is outstanding but is never a submitted GPU plan" {
  let scheduler = wb_service_scheduler()
  let service = wb_maintenance_service(
    scheduler,
    flight=FlightEvicting(15UL, 0UL),
    lifecycle=ServiceReady,
  )
  assert_eq(service.raw_outstanding_count(), 1)
  try service.flight_sequence_value() catch {
    Invalid(Sequence) => ()
    e => raise e
  } noraise {
    _ => fail("eviction fabricated a plan sequence")
  }
  let sequence = scheduler.last_submitted_plan_sequence()
  // This fixture deliberately has no process. It must fail on that exact
  // transport, not construct a successor while retirement is unacknowledged.
  try service.progress_impl() catch {
    Invalid(Lifecycle) => ()
    e => raise e
  } noraise {
    _ => fail("eviction was silently skipped")
  }
  assert_true(service.flight is FlightEvicting(15UL, 0UL))
  assert_eq(scheduler.last_submitted_plan_sequence(), sequence)
  service.clear_flight()
  assert_eq(service.raw_outstanding_count(), 0)
}

///|
test "terminal eviction waits for ACK and leaves logical lane ownership intact" {
  let lease = wb_terminal_online_lease()
  let generation = lease.lane_cut_source_generations[0]
  assert_eq(lease.next_device_retirement_lane(), 0)
  lease.device_retirement_lane = 0
  assert_eq(lease.next_device_retirement_lane(), 0)
  lease.lane_device_retired[0] = true
  lease.device_retirement_lane = -1
  assert_eq(lease.next_device_retirement_lane(), -1)
  assert_eq(lease.lane_phases[0], ONLINE_LANE_TERMINAL)
  assert_eq(lease.lane_cut_source_generations[0], generation)
  assert_eq(lease.active_lane_count, 1)
}

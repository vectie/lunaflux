///|
fn wb_terminal_online_lease() -> OnlineWorkerLease {
  let scheduler = wb_service_scheduler()
  let prepared = scheduler.prepare_admission(wb_online_request())
  let semantic_retention = @inference.LunaRequestStopTokenRetentionSlot::new()
  try! prepared.retain_stop_tokens_into(semantic_retention)
  try! scheduler.commit_admission(prepared, 0UL)
  let handle = try! prepared.request_handle()
  let terminal_generation = try! scheduler.cancel_exact(handle)
  ignore(try! scheduler.take_next_publication())
  if !scheduler.is_quiescent() || scheduler.is_pristine() {
    abort("terminal retirement scheduler fixture invariant")
  }
  let service : WorkerService = {
    scheduler,
    binding: wb_service_binding(),
    process: None,
    flight: FlightEmpty,
    lifecycle: ServiceReady,
    production_owned: true,
    ownership_mode: OnlineActiveOwnership,
    online_epoch: 7UL,
    online_publication_cursor: 11UL,
  }
  {
    service,
    epoch: 7UL,
    clock: @monotonic_clock.MonotonicClock::system(),
    request_phase: OnlineRequestTerminal,
    prepared_admission: None,
    semantic_retention,
    prepared_clock_read: @monotonic_clock.MonotonicClock::system().prepare_read(),
    handle: Some(handle),
    request_generation: terminal_generation,
    cut_source_generation: handle.request_generation(),
    publication_cursor: 11UL,
    expected_position: 3,
    cancel_reserved: false,
  }
}

///|
fn assert_terminal_retirement_rejected(
  lease : OnlineWorkerLease,
  has_exchange_capacity : Bool,
) -> Unit {
  try
    lease.preflight_terminal_request_retirement(
      lease.service,
      has_exchange_capacity,
    )
  catch {
    Invalid(Lifecycle) => ()
    _ => abort("unexpected terminal request retirement error")
  } noraise {
    _ => abort("invalid terminal request retirement was accepted")
  }
}

///|
fn wb_retirement_request(
  request_value : UInt64,
  received_at : UInt64,
) -> @core.TokenizedRequest {
  let limits = wb_service_inference_limits()
  try! @core.TokenizedRequest::new(
    request_id=try! @inference.RequestId::new(request_value),
    model_identity=wb_service_identity(),
    input_tokens=try! @inference.TokenBuffer::new([1], limits),
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=@inference.SamplingSeed::new(request_value),
    stop_tokens=wb_service_stop_token_projection(limits),
    stream_preference=Streaming,
    effective_limits=try! @inference.EffectiveLimits::new(
      max_new_tokens=2,
      context_ceiling=4,
      limits,
    ),
    admission_deadline=try! @inference.AdmissionDeadline::from_receipt(
      received_at,
      try! @inference.DeadlineBudget::from_milliseconds(1000UL, limits),
    ),
  )
}

///|
fn wb_empty_online_lease(
  clock : @monotonic_clock.MonotonicClock,
) -> OnlineWorkerLease {
  let service : WorkerService = {
    scheduler: wb_service_scheduler(),
    binding: wb_service_binding(),
    process: None,
    flight: FlightEmpty,
    lifecycle: ServiceReady,
    production_owned: true,
    ownership_mode: OnlineActiveOwnership,
    online_epoch: 17UL,
    online_publication_cursor: 0UL,
  }
  {
    service,
    epoch: 17UL,
    clock,
    request_phase: OnlineRequestEmpty,
    prepared_admission: None,
    semantic_retention: @inference.LunaRequestStopTokenRetentionSlot::new(),
    prepared_clock_read: clock.prepare_read(),
    handle: None,
    request_generation: @inference.RequestGeneration::initial(),
    cut_source_generation: @inference.RequestGeneration::initial(),
    publication_cursor: 0UL,
    expected_position: 0,
    cancel_reserved: false,
  }
}

///|
fn wb_retire_terminal_request(lease : OnlineWorkerLease) -> Unit {
  // The public method supplies this final scalar from the retained process.
  // White-box fixtures have no process owner, so drive its exact private
  // preflight and commit with positive exchange-capacity evidence.
  try! lease.preflight_terminal_request_retirement(lease.service, true)
  lease.commit_terminal_request_retirement()
}

///|
test "terminal request retirement resets only request-local state" {
  let lease = wb_terminal_online_lease()
  let service = lease.service
  let epoch = lease.epoch
  let publication_cursor = lease.publication_cursor
  let service_publication_cursor = service.online_publication_cursor
  let ownership = service.ownership_mode
  let generation_cursor = service.scheduler.status(
    try! @inference.RequestId::new(991UL),
  )
  lease.preflight_terminal_request_retirement(service, true)
  lease.commit_terminal_request_retirement()
  assert_true(lease.request_phase == OnlineRequestEmpty)
  assert_true(lease.handle is None)
  assert_eq(lease.request_generation, @inference.RequestGeneration::initial())
  assert_eq(
    lease.cut_source_generation,
    @inference.RequestGeneration::initial(),
  )
  assert_eq(lease.expected_position, 0)
  assert_false(lease.cancel_reserved)
  assert_eq(lease.epoch, epoch)
  assert_eq(lease.publication_cursor, publication_cursor)
  assert_eq(service.online_publication_cursor, service_publication_cursor)
  assert_true(service.ownership_mode == ownership)
  assert_eq(service.lifecycle, ServiceReady)
  assert_true(service.flight is FlightEmpty)
  assert_true(service.scheduler.is_quiescent())
  assert_false(service.scheduler.is_pristine())
  assert_eq(
    service.scheduler.status(try! @inference.RequestId::new(991UL)),
    generation_cursor,
  )
}

///|
test "terminal retirement preserves monotonic publication history across readmission" {
  let clock = @monotonic_clock.MonotonicClock::system()
  let lease = wb_empty_online_lease(clock)
  let service = lease.service
  let lease_epoch = lease.epoch

  let first_received_at = try! clock.now_millis()
  try! lease.admit(wb_retirement_request(1201UL, first_received_at))
  let first_handle = lease.handle.unwrap()
  let first_generation = first_handle.request_generation()
  assert_eq(try! lease.reserve_cancel(), OnlineCancelReserved)
  assert_eq(try! lease.commit_cancel(), OnlineCancelCommitted)
  let first_terminal = try! lease.take_publication()
  assert_eq(first_terminal.kind(), OnlineTerminalPublication)
  assert_eq(first_terminal.publication_sequence(), 1UL)
  assert_true(lease.request_phase == OnlineRequestTerminal)
  assert_eq(lease.publication_cursor, 1UL)
  assert_eq(service.online_publication_cursor, 1UL)

  wb_retire_terminal_request(lease)
  assert_true(lease.request_phase == OnlineRequestEmpty)
  assert_eq(lease.epoch, lease_epoch)
  assert_eq(lease.publication_cursor, 1UL)
  assert_eq(service.online_publication_cursor, 1UL)

  let second_received_at = try! clock.now_millis()
  try! lease.admit(wb_retirement_request(1202UL, second_received_at))
  let second_handle = lease.handle.unwrap()
  assert_true(second_handle != first_handle)
  assert_true(second_handle.request_id() != first_handle.request_id())
  assert_true(second_handle.request_generation() != first_generation)
  assert_true(
    second_handle.request_generation().value() > first_generation.value(),
  )
  assert_eq(lease.epoch, lease_epoch)
  assert_eq(try! lease.reserve_cancel(), OnlineCancelReserved)
  assert_eq(try! lease.commit_cancel(), OnlineCancelCommitted)
  let second_terminal = try! lease.take_publication()
  assert_eq(second_terminal.kind(), OnlineTerminalPublication)
  assert_eq(second_terminal.publication_sequence(), 2UL)
  assert_eq(lease.publication_cursor, 2UL)
  assert_eq(service.online_publication_cursor, 2UL)
  assert_eq(lease.epoch, lease_epoch)
  assert_true(service.scheduler.is_quiescent())
  assert_false(service.scheduler.is_pristine())
}

///|
test "terminal request retirement preflights every retained obligation" {
  let wrong_phase = wb_terminal_online_lease()
  wrong_phase.request_phase = OnlineRequestLive
  assert_terminal_retirement_rejected(wrong_phase, true)

  let reserved = wb_terminal_online_lease()
  reserved.cancel_reserved = true
  assert_terminal_retirement_rejected(reserved, true)

  let prepared = wb_terminal_online_lease()
  let other = wb_service_scheduler()
  prepared.prepared_admission = Some(
    try! other.prepare_exclusive_admission(wb_online_request()),
  )
  assert_terminal_retirement_rejected(prepared, true)

  let recovering = wb_terminal_online_lease()
  recovering.service.lifecycle = ServiceRecovering
  assert_terminal_retirement_rejected(recovering, true)

  let in_flight = wb_terminal_online_lease()
  in_flight.service.flight = FlightPlanned(9UL)
  assert_terminal_retirement_rejected(in_flight, true)

  let nonquiescent = wb_terminal_online_lease()
  ignore(try! nonquiescent.service.scheduler.admit(wb_online_request(), 0UL))
  assert_terminal_retirement_rejected(nonquiescent, true)

  let no_process_credit = wb_terminal_online_lease()
  assert_terminal_retirement_rejected(no_process_credit, false)
}

///|
test "terminal request retirement authenticates the current lease epoch first" {
  let lease = wb_terminal_online_lease()
  lease.epoch += 1UL
  try lease.require_current() catch {
    Invalid(OwnershipMode) => ()
    _ => fail("unexpected stale online lease error")
  } noraise {
    _ => fail("stale online lease authenticated")
  }
}

///|
test "terminal request retirement public symbol remains linked" {
  let _ = OnlineWorkerLease::retire_terminal_request
}

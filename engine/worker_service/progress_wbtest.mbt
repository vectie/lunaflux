///|
test "release allocation gate retains service progress symbols" {
  let _ = WorkerService::progress
  let _ = WorkerService::commit_received
  assert_eq(Bytes::make(17, b'Q').length(), 17)
  assert_eq(owned_preparation_allocation_positive_control().length(), 19)
}

///|
fn online_lease_allocation_positive_control() -> Bytes {
  Bytes::make(23, b'O')
}

///|
test "online lease allocation gate retains hot transition symbols" {
  let _ = OnlineWorkerLease::take_publication
  let _ = OnlineWorkerLease::take_reserved_token
  let _ = OnlineWorkerLease::commit_cancel
  let _ = OnlineWorkerLease::progress
  assert_eq(online_lease_allocation_positive_control().length(), 23)
}

///|
test "natural terminal wins before online monotonic clock sampling" {
  let reads = Ref(0)
  let result = read_expiry_clock(true, reads, fn(
    counter,
  ) raise WorkerServiceError {
    counter.val += 1
    raise Invalid(Clock)
  })
  assert_true(result is ExpiryNaturalTerminal)
  assert_eq(reads.val, 0)
}

///|
test "online deadline clock failure samples once before scheduler mutation" {
  let reads = Ref(0)
  try
    read_expiry_clock(false, reads, fn(counter) raise WorkerServiceError {
      counter.val += 1
      raise Invalid(Clock)
    })
  catch {
    Invalid(Clock) => ()
    _ => fail("unexpected online deadline clock error")
  } noraise {
    _ => fail("failing online deadline clock returned a sample")
  }
  assert_eq(reads.val, 1)
}

///|
fn owned_preparation_allocation_positive_control() -> Bytes {
  Bytes::make(19, b'R')
}

///|
fn wb_online_request(deadline? : UInt64 = 100UL) -> @core.TokenizedRequest {
  let limits = wb_service_inference_limits()
  try! @core.TokenizedRequest::new(
    request_id=try! @inference.RequestId::new(991UL),
    model_identity=wb_service_identity(),
    input_tokens=try! @inference.TokenBuffer::new([1], limits),
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=@inference.SamplingSeed::new(991UL),
    stops=try! @inference.StopConditions::new([], [], limits),
    cache=@inference.CachePolicy::new(
      try! @inference.CacheScope::from_ascii(b"online", limits),
      Disabled,
    ),
    stream_preference=Streaming,
    effective_limits=try! @inference.EffectiveLimits::new(
      max_new_tokens=2,
      context_ceiling=4,
      limits,
    ),
    admission_deadline=try! @inference.AdmissionDeadline::from_receipt(
      0UL,
      try! @inference.DeadlineBudget::from_milliseconds(deadline, limits),
    ),
  )
}

///|
fn wb_prepared_online_lease(
  scheduler : @core.Scheduler,
  prepared : @core.PreparedExclusiveAdmission,
  prepared_clock_read : @monotonic_clock.PreparedMonotonicRead,
) -> OnlineWorkerLease {
  let service : WorkerService = {
    scheduler,
    binding: wb_service_binding(),
    process: None,
    flight_state: FLIGHT_EMPTY,
    flight_sequence: 0UL,
    lifecycle: ServiceReady,
    production_owned: true,
    ownership_mode: SERVICE_MODE_ONLINE_ACTIVE,
    online_epoch: 1UL,
    online_publication_cursor: 0UL,
  }
  {
    service,
    epoch: 1UL,
    clock: @monotonic_clock.MonotonicClock::system(),
    request_phase: ONLINE_REQUEST_EMPTY,
    prepared_admission: Some(prepared),
    prepared_clock_read,
    handle: None,
    request_generation: @inference.RequestGeneration::initial(),
    cut_source_generation: @inference.RequestGeneration::initial(),
    publication_cursor: 0UL,
    expected_position: 0,
    cancel_reserved: false,
  }
}

///|
test "prepared online clock failure aborts exclusive admission without mutation" {
  let scheduler = wb_service_scheduler()
  let prepared = try! scheduler.prepare_exclusive_admission(wb_online_request())
  let read = @monotonic_clock.MonotonicClock::system().prepare_read()
  assert_eq(read.read_status(), MonotonicReadReady)
  let lease = wb_prepared_online_lease(scheduler, prepared, read)
  assert_eq(lease.commit_prepared_admission(), OnlineAdmissionClockFailed)
  assert_true(scheduler.is_pristine())
  assert_eq(scheduler.live_request_count(), 0)
  assert_true(lease.prepared_admission is None)
  assert_eq(
    scheduler.commit_exclusive_admission(prepared, 0UL),
    ExclusiveAdmissionInvalidated,
  )
}

///|
test "prepared online expiry clears reservation and retains empty cleanup authority" {
  let scheduler = wb_service_scheduler()
  let prepared = try! scheduler.prepare_exclusive_admission(
    wb_online_request(deadline=1UL),
  )
  let read = @monotonic_clock.MonotonicClock::system().prepare_read()
  let lease = wb_prepared_online_lease(scheduler, prepared, read)
  assert_eq(lease.commit_prepared_admission(), OnlineAdmissionExpired)
  assert_true(scheduler.is_pristine())
  assert_eq(scheduler.live_request_count(), 0)
  assert_true(lease.prepared_admission is None)
  assert_eq(lease.request_phase, ONLINE_REQUEST_EMPTY)
}

///|
test "online restart-forbidden drain authenticates exact request" {
  let scheduler = wb_service_scheduler()
  let handle = try! scheduler.admit(wb_online_request(), 0UL)
  let service : WorkerService = {
    scheduler,
    binding: wb_service_binding(),
    process: None,
    flight_state: FLIGHT_EMPTY,
    flight_sequence: 0UL,
    lifecycle: ServiceRestartForbidden,
    production_owned: true,
    ownership_mode: SERVICE_MODE_ONLINE_ACTIVE,
    online_epoch: 1UL,
    online_publication_cursor: 0UL,
  }
  let lease : OnlineWorkerLease = {
    service,
    epoch: 1UL,
    clock: @monotonic_clock.MonotonicClock::system(),
    request_phase: ONLINE_REQUEST_LIVE,
    prepared_admission: None,
    prepared_clock_read: @monotonic_clock.MonotonicClock::system().prepare_read(),
    handle: Some(handle),
    request_generation: handle.request_generation(),
    cut_source_generation: @inference.RequestGeneration::initial(),
    publication_cursor: 0UL,
    expected_position: 0,
    cancel_reserved: false,
  }
  assert_eq(lease.drain_restart_forbidden(), Complete)
  assert_eq(service.lifecycle(), ServiceTerminalCloseReady)
  let terminal = lease.take_publication()
  assert_eq(terminal.terminal_reason(), OnlineWorkerFailed)
}

///|
test "online terminal recovery wrapper drains restart-forbidden request" {
  let scheduler = wb_service_scheduler()
  let handle = try! scheduler.admit(wb_online_request(), 0UL)
  let service : WorkerService = {
    scheduler,
    binding: wb_service_binding(),
    process: None,
    flight_state: FLIGHT_EMPTY,
    flight_sequence: 0UL,
    lifecycle: ServiceRestartForbidden,
    production_owned: true,
    ownership_mode: SERVICE_MODE_ONLINE_ACTIVE,
    online_epoch: 1UL,
    online_publication_cursor: 0UL,
  }
  let lease : OnlineWorkerLease = {
    service,
    epoch: 1UL,
    clock: @monotonic_clock.MonotonicClock::system(),
    request_phase: ONLINE_REQUEST_LIVE,
    prepared_admission: None,
    prepared_clock_read: @monotonic_clock.MonotonicClock::system().prepare_read(),
    handle: Some(handle),
    request_generation: handle.request_generation(),
    cut_source_generation: @inference.RequestGeneration::initial(),
    publication_cursor: 0UL,
    expected_position: 0,
    cancel_reserved: false,
  }
  assert_eq(
    lease.progress_terminal_recovery(ResourceUnavailable),
    OnlineTerminalRecoveryPending,
  )
  assert_eq(service.lifecycle(), ServiceTerminalCloseReady)
  assert_eq(
    lease.progress_terminal_recovery(ResourceUnavailable),
    OnlineTerminalRecoveryPublicationReady,
  )
  assert_eq(lease.take_publication().terminal_reason(), OnlineWorkerFailed)
}

///|
test "online worker step classification is bounded and state-aware" {
  assert_eq(
    classify_online_worker_error(
      Scheduler(Invalid(field=OutputCapacity, issue=Backpressured)),
    ),
    0,
  )
  assert_eq(
    classify_online_worker_error(
      RootBoundProcess(RootBoundFailure(Process(Backpressured))),
    ),
    0,
  )
  assert_eq(
    classify_online_worker_error(
      RootBoundProcess(RootBoundFailure(Process(Invalid(Completion)))),
    ),
    1,
  )
  assert_eq(
    classify_online_worker_error(
      Scheduler(Invalid(field=Completion, issue=Mismatch)),
    ),
    1,
  )
  assert_eq(
    classify_online_worker_error(
      RootBoundProcess(RootBoundFailure(Process(Transport(ChannelClosed)))),
    ),
    2,
  )
  assert_eq(classify_online_worker_error(Invalid(Lifecycle)), 2)
  assert_eq(
    classify_online_worker_error(Scheduler(PageAllocator(Exhausted))),
    2,
  )
  assert_eq(classify_online_worker_error(Scheduler(BlockTable(Exhausted))), 2)
  assert_eq(
    classify_online_worker_error(
      Scheduler(Invalid(field=PublicationSequence, issue=Exhausted)),
    ),
    2,
  )
}

///|
test "out-of-phase reserved dequeue preserves scheduler publication" {
  let scheduler = wb_service_scheduler()
  let admitted = try! scheduler.admit(wb_online_request(), 0UL)
  let outcome = try! scheduler.build_next()
  let plan = try! scheduler.submitted_plan(outcome)
  let row = try! plan.prefill_row(0)
  let writer = try! scheduler.completion_writer(plan)
  try! writer.append_final_prefill_completed(
    slot=try! row.completion_slot(),
    request_id=try! row.request_id(),
    request_generation=try! row.request_generation(),
    processed_tokens=try! row.tokens().length(),
    token_id=7,
  )
  try! scheduler.complete(plan, try! writer.submit())
  let handle = scheduler.status(admitted.request_id()).unwrap().handle()
  assert_eq(try! scheduler.reserve_cancel_exact(handle), CancelReserved)
  let service : WorkerService = {
    scheduler,
    binding: wb_service_binding(),
    process: None,
    flight_state: FLIGHT_EMPTY,
    flight_sequence: 0UL,
    lifecycle: ServiceReady,
    production_owned: true,
    ownership_mode: SERVICE_MODE_ONLINE_ACTIVE,
    online_epoch: 1UL,
    online_publication_cursor: 0UL,
  }
  let lease : OnlineWorkerLease = {
    service,
    epoch: 1UL,
    clock: @monotonic_clock.MonotonicClock::system(),
    request_phase: ONLINE_REQUEST_TERMINATION_PENDING,
    prepared_admission: None,
    prepared_clock_read: @monotonic_clock.MonotonicClock::system().prepare_read(),
    handle: Some(handle),
    request_generation: handle.request_generation(),
    cut_source_generation: admitted.request_generation(),
    publication_cursor: 0UL,
    expected_position: 0,
    cancel_reserved: true,
  }
  try lease.take_reserved_token() catch {
    Invalid(Lifecycle) => ()
    _ => fail("unexpected out-of-phase reserved dequeue error")
  } noraise {
    _ => fail("out-of-phase lease dequeued reserved publication")
  }
  assert_eq(scheduler.generated_token_count(), 1)
  assert_eq(try! scheduler.take_reserved_generated_token().token_id(), 7)
  scheduler.abort_cancel_reservation()
}

///|
test "terminal close retry retains shutdown-stage and close-stage authority" {
  assert_eq(terminal_close_retry_action(ServiceRecovering), 1)
  assert_eq(terminal_close_retry_action(ServiceClosingFailed), 2)
  try terminal_close_retry_action(ServiceReady) catch {
    Invalid(Lifecycle) => ()
    _ => fail("unexpected terminal close retry phase error")
  } noraise {
    _ => fail("healthy service entered terminal close retry")
  }
}

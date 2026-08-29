///|
fn wb_multi_worker_limits() -> @worker_protocol.WorkerProtocolLimits {
  try! @worker_protocol.WorkerProtocolLimits::new(
    max_prefill_rows=2,
    max_decode_rows=2,
    max_plan_rows=2,
    max_plan_tokens=2,
    max_plan_pages=4,
    max_capabilities=4,
    max_completion_slots=2,
    max_sequence_tokens=4,
    max_token_id=31,
  )
}

///|
fn wb_multi_scheduler() -> @core.Scheduler {
  let worker = wb_multi_worker_limits()
  let scheduling = try! @scheduler_config.new(
    step_token_budget=2,
    max_active_requests=2,
    max_waiting_requests=2,
    prefill_chunk_tokens=1,
    emergency_decode_page_reserve=1,
    waiting_age_threshold_steps=2,
    output_event_capacity=4,
  )
  let cache = try! @cache.new(
    tokens_per_page=2,
    total_page_count=8,
    block_table_pages_per_request=2,
    prefix_enabled=false,
    max_prefix_entries=0,
    max_prefix_nodes=0,
    max_prefix_tokens_per_entry=0,
    max_prefix_pages=0,
    max_active_references_per_page=4,
    max_cached_references_per_page=4,
    layout_version=1,
  )
  let shapes = try! @plan.ShapeConstraints::new(
    max_batch_rows=2,
    max_sequence_tokens=4,
    vocabulary_size=32,
    hidden_size=8,
    intermediate_size=16,
    attention_head_count=2,
    kv_head_count=1,
    head_dimension=4,
  )
  let capacity = try! @runtime_resolved.resolve(
    scheduling, cache, shapes, worker,
  )
  let generation = @worker_protocol.ModelPlanGeneration::first()
  let recipe = try! @worker_protocol.RowCapabilityRecipe::new(
    identity=wb_service_identity(),
    model_plan_generation=generation,
    [@plan.KernelCapabilityId::embedding_lookup()],
    worker,
  )
  try! @core.Scheduler::new(
    capacity,
    wb_service_identity(),
    generation,
    recipe,
    recipe,
    recipe,
  )
}

///|
fn wb_multi_request(
  value : UInt64,
  received_at? : UInt64 = 0UL,
) -> @core.TokenizedRequest {
  let limits = wb_service_inference_limits()
  try! @core.TokenizedRequest::new(
    request_id=try! @inference.RequestId::new(value),
    model_identity=wb_service_identity(),
    input_tokens=try! @inference.TokenBuffer::new([1], limits),
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=@inference.SamplingSeed::new(value),
    stop_tokens=wb_service_stop_token_projection(limits),
    stream_preference=Streaming,
    effective_limits=try! @inference.EffectiveLimits::new(
      max_new_tokens=2,
      context_ceiling=4,
      limits,
    ),
    admission_deadline=try! @inference.AdmissionDeadline::from_receipt(
      received_at,
      try! @inference.DeadlineBudget::from_milliseconds(100UL, limits),
    ),
  )
}

///|
test "fixed online lanes admit two requests into one mixed worker plan" {
  let scheduler = wb_multi_scheduler()
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
  let clock = @monotonic_clock.MonotonicClock::system()
  let received_at = try! clock.now_millis()
  let lease = wb_online_lane_lease(
    service,
    clock,
    None,
    clock.prepare_read(),
    None,
    ONLINE_LANE_EMPTY,
    @inference.RequestGeneration::initial(),
    @inference.RequestGeneration::initial(),
    0,
    0,
    false,
    @inference.LunaRequestStopTokenRetentionSlot::new(),
    lane_count=2,
  )
  let admitted_a = try! lease.try_admit_request(
    wb_multi_request(2001UL, received_at~),
  )
  let admitted_b = try! lease.try_admit_request(
    wb_multi_request(2002UL, received_at~),
  )
  assert_eq(admitted_a.kind(), LunaOnlineWorkerAdmitted)
  assert_eq(admitted_b.kind(), LunaOnlineWorkerAdmitted)
  let request_a = try! admitted_a.request()
  let request_b = try! admitted_b.request()
  assert_true(request_a.is_live())
  assert_true(request_b.is_live())
  assert_eq(try! lease.active_request_count(), 2)
  assert_eq(
    (try! lease.try_admit_request(wb_multi_request(2003UL, received_at~))).kind(),
    LunaOnlineWorkerSaturated,
  )
  let outcome = try! scheduler.build_next()
  let plan = try! scheduler.submitted_plan(outcome)
  assert_eq(try! plan.row_count(), 2)
  assert_eq(try! plan.token_count(), 2)
}

///|
test "multi-lane restart terminalizes waiting requests in global order" {
  let scheduler = wb_multi_scheduler()
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
  let clock = @monotonic_clock.MonotonicClock::system()
  let received_at = try! clock.now_millis()
  let lease = wb_online_lane_lease(
    service,
    clock,
    None,
    clock.prepare_read(),
    None,
    ONLINE_LANE_EMPTY,
    @inference.RequestGeneration::initial(),
    @inference.RequestGeneration::initial(),
    0,
    0,
    false,
    @inference.LunaRequestStopTokenRetentionSlot::new(),
    lane_count=2,
  )
  let request_a = try! (try! lease.try_admit_request(
    wb_multi_request(2101UL, received_at~),
  )).request()
  let request_b = try! (try! lease.try_admit_request(
    wb_multi_request(2102UL, received_at~),
  )).request()
  service.lifecycle = ServiceDeviceInvalidationRequired
  assert_eq(try! lease.invalidate_device_state(), 0)
  assert_eq(service.lifecycle, ServiceRestartReady)
  service.lifecycle = ServiceReady
  assert_false(try! lease.progress_restarted_failure_terminals())
  assert_true(try! lease.progress_restarted_failure_terminals())
  let first = try! lease.take_publication()
  let second = try! lease.take_publication()
  assert_eq(first.publication_sequence(), 1UL)
  assert_eq(second.publication_sequence(), 2UL)
  assert_eq(first.terminal_reason(), OnlineWorkerFailed)
  assert_eq(second.terminal_reason(), OnlineWorkerFailed)
  assert_eq(first.route().lane(), 0)
  assert_eq(second.route().lane(), 1)
  try! request_a.retire_terminal()
  try! request_b.retire_terminal()
  assert_eq(try! lease.active_request_count(), 0)
  assert_false(request_a.is_live())
  assert_false(request_b.is_live())
}

///|
test "terminal publication alone carries request-local cached usage" {
  let terminal = OnlineWorkerPublication(1, 4, 3, 9UL, 1, 7UL)
  assert_eq(try! terminal.cached_input_tokens(), 4)
  assert_eq(try! terminal.terminal_reason(), OnlineMaximumOutputReached)

  let token = OnlineWorkerPublication(0, 17, 0, 8UL, 0, 6UL)
  try token.cached_input_tokens() catch {
    Invalid(Sequence) => ()
    _ => fail("unexpected token cached-usage error")
  } noraise {
    _ => fail("token publication exposed terminal cached usage")
  }

  let suppressed = OnlineWorkerPublication(2, 0, 0, 8UL, 0, 6UL)
  try suppressed.cached_input_tokens() catch {
    Invalid(Sequence) => ()
    _ => fail("unexpected suppressed cached-usage error")
  } noraise {
    _ => fail("suppressed publication exposed terminal cached usage")
  }
}

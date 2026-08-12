///|
fn wb_identity() -> @spec.ModelIdentity {
  @spec.ModelIdentity::new(
    try! @spec.ContentDigest::from_sha256(
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
    try! @spec.PlanDigest::from_sha256(
      "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
    ),
  )
}

///|
fn wb_request(id : UInt64, deadline? : UInt64 = 100UL) -> TokenizedRequest {
  let limits = try! @inference.InferenceLimits::new(
    max_text_bytes=64,
    max_input_tokens=8,
    max_new_tokens=4,
    max_context_tokens=16,
    max_token_id=31,
    max_stop_token_ids=4,
    max_stop_strings=2,
    max_stop_string_bytes=8,
    max_trace_bytes=8,
    max_cache_scope_bytes=8,
    max_decoded_delta_bytes=8,
    max_deadline_millis=1000UL,
    max_top_k=32,
    max_temperature=4.0,
  )
  try! TokenizedRequest::new(
    request_id=try! @inference.RequestId::new(id),
    model_identity=wb_identity(),
    input_tokens=try! @inference.TokenBuffer::new([1, 2], limits),
    sampling=@inference.SamplingParameters::greedy(),
    stops=try! @inference.StopConditions::new([], [], limits),
    cache=@inference.CachePolicy::new(
      try! @inference.CacheScope::from_ascii(b"scope", limits),
      Disabled,
    ),
    stream_preference=Streaming,
    effective_limits=try! @inference.EffectiveLimits::new(
      max_new_tokens=2,
      context_ceiling=8,
      limits,
    ),
    deadline_budget=try! @inference.DeadlineBudget::from_milliseconds(
      deadline, limits,
    ),
  )
}

///|
fn wb_owner() -> Scheduler {
  let scheduler = try! @scheduler.new(
    step_token_budget=2,
    max_active_requests=2,
    max_waiting_requests=2,
    prefill_chunk_tokens=2,
    emergency_decode_page_reserve=1,
    waiting_age_threshold_steps=2,
    output_event_capacity=2,
  )
  let cache = try! @cache.new(
    tokens_per_page=4,
    total_page_count=8,
    block_table_pages_per_request=4,
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
    max_sequence_tokens=16,
    vocabulary_size=32,
    hidden_size=8,
    intermediate_size=16,
    attention_head_count=2,
    kv_head_count=1,
    head_dimension=4,
  )
  let worker = try! @worker_protocol.WorkerProtocolLimits::new(
    max_prefill_rows=2,
    max_decode_rows=2,
    max_plan_rows=2,
    max_plan_tokens=2,
    max_plan_pages=8,
    max_capabilities=8,
    max_completion_slots=2,
    max_sequence_tokens=16,
    max_token_id=31,
  )
  let capacity = try! @runtime_resolved.resolve(
    scheduler, cache, shapes, worker,
  )
  let generation = @worker_protocol.ModelPlanGeneration::first()
  let recipe = try! @worker_protocol.RowCapabilityRecipe::new(
    identity=wb_identity(),
    model_plan_generation=generation,
    [@plan.KernelCapabilityId::embedding_lookup()],
    worker,
  )
  try! Scheduler::new(capacity, wb_identity(), generation, recipe, recipe)
}

///|
test "generation exhaustion rejects admission before consuming a slot" {
  let owner = wb_owner()
  owner.generation_cursor = MAX_GENERATION
  try owner.admit(wb_request(1UL), 0UL) catch {
    error => assert_eq(error, Invalid(field=RequestGeneration, issue=Exhausted))
  } noraise {
    _ => fail("expected generation exhaustion")
  }
  assert_eq(owner.live_count, 0)
  assert_eq(owner.waiting_count, 0)
  assert_eq(owner.free_head, 0)
  assert_eq(owner.terminal_count, 0)
}

///|
test "generation exhaustion rejects cancel without partial mutation" {
  let owner = wb_owner()
  let handle = owner.admit(wb_request(1UL), 0UL)
  let waiting_head = owner.waiting_head
  let waiting_tail = owner.waiting_tail
  let slot_state = owner.slot_states[handle.slot().index()]
  owner.generation_cursor = MAX_GENERATION
  try owner.cancel(handle) catch {
    error => assert_eq(error, Invalid(field=RequestGeneration, issue=Exhausted))
  } noraise {
    _ => fail("expected cancel generation exhaustion")
  }
  let status = owner.status(handle.request_id()).unwrap()
  assert_eq(status.handle(), handle)
  assert_eq(status.state(), Waiting)
  assert_eq(owner.slot_states[handle.slot().index()], slot_state)
  assert_eq(owner.waiting_head, waiting_head)
  assert_eq(owner.waiting_tail, waiting_tail)
  assert_eq(owner.waiting_count, 1)
  assert_eq(owner.live_count, 1)
  assert_eq(owner.terminal_count, 0)
}

///|
test "deadline generation preflight prevents partial expiry" {
  let owner = wb_owner()
  let first = owner.admit(wb_request(1UL, deadline=10UL), 0UL)
  let second = owner.admit(wb_request(2UL, deadline=10UL), 0UL)
  owner.generation_cursor = MAX_GENERATION - 1UL
  try owner.expire_deadlines(10UL) catch {
    error => assert_eq(error, Invalid(field=RequestGeneration, issue=Exhausted))
  } noraise {
    _ => fail("expected deadline generation exhaustion")
  }
  assert_eq(owner.status(first.request_id()).unwrap().handle(), first)
  assert_eq(owner.status(second.request_id()).unwrap().handle(), second)
  assert_eq(owner.waiting_count, 2)
  assert_eq(owner.live_count, 2)
  assert_eq(owner.terminal_count, 0)
}

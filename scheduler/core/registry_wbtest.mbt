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
  wb_owner_with_capacity(2, 2)
}

///|
fn wb_owner_with_capacity(
  max_active_requests : Int,
  max_waiting_requests : Int,
) -> Scheduler {
  let scheduler = try! @scheduler.new(
    step_token_budget=2,
    max_active_requests~,
    max_waiting_requests~,
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
  try! Scheduler::new(
    capacity,
    wb_identity(),
    generation,
    recipe,
    recipe,
    recipe,
  )
}

///|
fn wb_next_plan(owner : Scheduler) -> @worker_protocol.SubmittedSchedulePlan {
  try! owner.submitted_plan(try! owner.build_next())
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

///|
test "distinct plan owners permit two different in-flight requests" {
  let owner = wb_owner_with_capacity(3, 3)
  let first = owner.admit(wb_request(11UL), 0UL)
  let second = owner.admit(wb_request(12UL), 0UL)
  let plan_a = wb_next_plan(owner)
  assert_eq(try! plan_a.row_count(), 1)
  assert_eq(try! plan_a.prefill_row(0).request_id(), first.request_id())
  let plan_b = wb_next_plan(owner)
  assert_eq(try! plan_b.row_count(), 1)
  assert_eq(try! plan_b.prefill_row(0).request_id(), second.request_id())
  assert_true(try! owner.build_next().is_backpressured())
  assert_eq(owner.plan_a_sequence, 1UL)
  assert_eq(owner.plan_b_sequence, 2UL)
  assert_eq(owner.plan_a_slots[0], first.slot().index())
  assert_eq(owner.plan_b_slots[0], second.slot().index())
  ignore(owner.admit(wb_request(13UL), 0UL))
  assert_true(try! owner.build_next().is_backpressured())
}

///|
test "checkpoint cleanup restores exact identities before a real build" {
  let owner = wb_owner()
  let handle = owner.admit(wb_request(41UL), 0UL)
  assert_eq(owner.select_rows(), 2)
  let table_checkpoint = try! owner.block_tables.checkpoint()
  let page_checkpoint = try! owner.page_allocator.checkpoint()
  let plan_checkpoint = try! owner.plan_a.checkpoint()
  let slot = owner.prefill_slots[0]
  let table = owner.row_table(slot)
  let old_count = try! owner.block_tables.page_count(table)
  owner.append_pages(table, old_count, 1, slot)
  let provisional_page = try! owner.block_tables.page_at(table, 0)
  try! owner.plan_a.rollback_to(plan_checkpoint)
  owner.rollback_tables(table_checkpoint)
  try! owner.block_tables.rollback_to(table_checkpoint)
  try! owner.page_allocator.rollback_to(page_checkpoint)
  assert_eq(owner.journal_count, 0)
  assert_eq(owner.waiting_count, 1)
  assert_eq(owner.status(handle.request_id()).unwrap().state(), Waiting)
  assert_eq(owner.free_page_count(), 8)
  assert_eq(owner.allocated_block_table_count(), 0)
  let plan = wb_next_plan(owner)
  assert_eq(try! plan.page_at(0), provisional_page)
  assert_eq(try! owner.request_tables.read(slot), table)
}

///|
fn make_decode_active(
  owner : Scheduler,
  handle : RequestHandle,
  output_count : Int,
  token : Int,
) -> Unit {
  let slot = handle.slot().index()
  owner.remove_waiting(slot)
  let table = try! owner.block_tables.allocate()
  let run = try! @page_allocator.PageRunBuffer::new(1)
  try! owner.page_allocator.allocate_run(1, run)
  try! owner.block_tables.append_run(table, run)
  try! owner.request_tables.write(slot, table)
  owner.slot_states[slot] = SLOT_DECODE
  owner.output_counts[slot] = output_count
  owner.decode_tokens[slot] = token
  owner.active_slots[owner.active_count] = slot
  owner.active_count += 1
}

///|
test "decode is reserved first while protocol serializes prefill first" {
  let owner = wb_owner()
  let decoding = owner.admit(wb_request(21UL), 0UL)
  make_decode_active(owner, decoding, 1, 7)
  let waiting = owner.admit(wb_request(22UL), 0UL)
  let plan = wb_next_plan(owner)
  assert_eq(try! plan.token_budget(), 2)
  assert_eq(try! plan.prefill_count(), 1)
  assert_eq(try! plan.decode_count(), 1)
  assert_eq(try! plan.prefill_row(0).request_id(), waiting.request_id())
  assert_eq(try! plan.decode_row(0).request_id(), decoding.request_id())
  assert_eq(try! plan.prefill_row(0).tokens().length(), 1)
  assert_eq(try! plan.decode_row(0).sequence_token_position(), 2)
  assert_eq(try! plan.decode_row(0).input_token().length(), 1)
  assert_eq(try! plan.token_at(1), 7)
}

///|
test "multirow second-row failure restores every provisional identity" {
  let owner = wb_owner_with_capacity(3, 3)
  let first = owner.admit(wb_request(51UL), 0UL)
  let second = owner.admit(wb_request(52UL), 0UL)
  make_decode_active(owner, first, 3, 7)
  make_decode_active(owner, second, 3, 99)
  let free_pages = owner.free_page_count()
  try owner.build_next() catch {
    WorkerProtocol(_) => ()
    _ => fail("unexpected multirow rollback error")
  } noraise {
    _ => fail("expected second-row failure")
  }
  assert_eq(owner.free_page_count(), free_pages)
  assert_eq(owner.plan_a.state(), Writable)
  assert_false(owner.status(first.request_id()).unwrap().in_flight())
  assert_false(owner.status(second.request_id()).unwrap().in_flight())
  owner.decode_tokens[second.slot().index()] = 8
  let plan = wb_next_plan(owner)
  assert_eq(try! plan.decode_count(), 2)
  let first_appended = try! plan.page_at(1)
  let second_appended = try! plan.page_at(3)
  assert_eq(first_appended.index(), 2)
  assert_eq(first_appended.generation(), 1)
  assert_eq(second_appended.index(), 3)
  assert_eq(second_appended.generation(), 1)
}

///|
test "post-allocation row failure restores exact page and table ownership" {
  let owner = wb_owner()
  let decoding = owner.admit(wb_request(31UL), 0UL)
  make_decode_active(owner, decoding, 3, 99)
  let free_pages = owner.free_page_count()
  let table_count = owner.allocated_block_table_count()
  let table = try! owner.request_tables.read(decoding.slot().index())
  let page_count = try! owner.block_tables.page_count(table)
  try owner.build_next() catch {
    WorkerProtocol(_) => ()
    _ => fail("unexpected rollback error")
  } noraise {
    _ => fail("expected invalid decode token")
  }
  assert_eq(owner.free_page_count(), free_pages)
  assert_eq(owner.allocated_block_table_count(), table_count)
  assert_eq(try! owner.block_tables.page_count(table), page_count)
  assert_eq(owner.status(decoding.request_id()).unwrap().state(), Decode)
  assert_false(owner.status(decoding.request_id()).unwrap().in_flight())
  assert_eq(owner.plan_a.state(), Writable)
  assert_eq(owner.plan_a_sequence, 0UL)
  assert_eq(owner.journal_count, 0)
  owner.decode_tokens[decoding.slot().index()] = 7
  let plan = wb_next_plan(owner)
  assert_eq(try! plan.page_count(), 2)
  let restored = try! plan.page_at(1)
  assert_eq(restored.index(), 1)
  assert_eq(restored.generation(), 1)
}

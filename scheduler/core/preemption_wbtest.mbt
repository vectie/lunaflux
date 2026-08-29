///|
fn wb_prefill_completion(
  owner : Scheduler,
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> @worker_protocol.SubmittedCompletion {
  let row = try! plan.prefill_row(0)
  let writer = try! owner.completion_writer(plan)
  try! writer.append_prefill_completed(
    slot=try! row.completion_slot(),
    request_id=try! row.request_id(),
    request_generation=try! row.request_generation(),
    processed_tokens=try! row.tokens().length(),
  )
  try! writer.submit()
}

///|
test "preemption releases KV and replay excludes the retained latest token" {
  let owner = wb_owner_with_capacity(1, 2)
  let active = owner.admit(wb_request(701UL), 0UL)
  make_decode_active(owner, active, 0, 0)
  let active_slot = active.slot().index()
  owner.processed_inputs[active_slot] = 2
  owner.prefill_cursors[active_slot] = 2
  try! owner.enqueue_generated_token(active_slot, 7)
  owner.store_generated_token(active_slot, 0, 7)
  owner.output_counts[active_slot] += 1
  try! owner.enqueue_generated_token(active_slot, 8)
  owner.store_generated_token(active_slot, 1, 8)
  owner.output_counts[active_slot] += 1
  owner.decode_tokens[active_slot] = 8
  let urgent = owner.admit(wb_request(702UL), 0UL)
  owner.step = 2UL
  owner.admitted_steps[urgent.slot().index()] = 0UL
  let generation = active.request_generation()
  let publication_cursor = owner.publication_cursor

  let outcome = try! owner.build_next()
  assert_true(outcome.is_preempted())
  assert_eq(
    owner.status(active.request_id()).unwrap().state(),
    RecomputeWaiting,
  )
  assert_eq(
    owner.status(active.request_id()).unwrap().processed_input_tokens(),
    2,
  )
  assert_eq(owner.free_page_count(), 8)
  assert_eq(owner.allocated_block_table_count(), 0)
  assert_eq(owner.last_submitted_plan_sequence(), 0UL)
  assert_eq(
    owner.status(active.request_id()).unwrap().handle().request_generation(),
    generation,
  )
  assert_eq(owner.publication_cursor, publication_cursor)
  assert_eq(owner.generated_token_count(), 2)
  assert_eq(try! owner.take_generated_token().token_id(), 7)
  assert_eq(try! owner.take_generated_token().token_id(), 8)

  ignore(owner.cancel(urgent))
  let first = wb_next_plan(owner)
  assert_eq(try! first.prefill_row(0).tokens().length(), 2)
  assert_eq(try! first.token_at(0), 1)
  assert_eq(try! first.token_at(1), 2)
  ignore(owner.complete(first, wb_prefill_completion(owner, first)))
  assert_eq(
    owner.status(active.request_id()).unwrap().state(),
    RecomputePrefill,
  )
  assert_eq(
    owner.status(active.request_id()).unwrap().processed_input_tokens(),
    2,
  )

  let second = wb_next_plan(owner)
  assert_eq(try! second.prefill_row(0).tokens().length(), 1)
  assert_eq(try! second.token_at(0), 7)
  ignore(owner.complete(second, wb_prefill_completion(owner, second)))
  assert_eq(owner.status(active.request_id()).unwrap().state(), Decode)
  assert_eq(owner.decode_tokens[active_slot], 8)
  assert_eq(
    owner.status(active.request_id()).unwrap().processed_input_tokens(),
    2,
  )
}

///|
test "ineligible aged prompt reserves nothing from a runnable decode" {
  let owner = wb_owner_with_capacity(2, 2)
  let decoding = owner.admit(wb_request(711UL), 0UL)
  make_decode_active(owner, decoding, 1, 7)
  let waiting = owner.admit(wb_request(712UL), 0UL)
  owner.step = 2UL
  owner.admitted_steps[waiting.slot().index()] = 0UL
  let held = try! @page_allocator.PageIdStorage::new(6)
  for index in 0..<6 {
    try! held.write(index, try! owner.page_allocator.allocate_one())
  }
  assert_eq(
    owner.free_page_count(),
    owner.capacity.emergency_decode_page_reserve(),
  )
  assert_eq(try! owner.select_rows(), 1)
  assert_eq(owner.prefill_count, 0)
  assert_eq(owner.decode_count, 1)
  assert_eq(owner.decode_slots[0], decoding.slot().index())
  for index in 0..<6 {
    try! owner.page_allocator.release_active(try! held.read(index))
  }
}

///|
test "decode selection cursor gives every ready slot a row" {
  let owner = wb_owner_with_capacity(3, 3)
  let first = owner.admit(wb_request(721UL), 0UL)
  let second = owner.admit(wb_request(722UL), 0UL)
  let third = owner.admit(wb_request(723UL), 0UL)
  make_decode_active(owner, first, 1, 7)
  make_decode_active(owner, second, 1, 8)
  make_decode_active(owner, third, 1, 9)

  assert_eq(try! owner.select_rows(), 2)
  assert_eq(owner.decode_slots[0], first.slot().index())
  assert_eq(owner.decode_slots[1], second.slot().index())
  assert_eq(owner.pending_decode_cursor, third.slot().index())
  owner.commit_plan_state(0)
  assert_eq(owner.decode_cursor, third.slot().index())
  owner.in_flight[first.slot().index()] = false
  owner.in_flight[second.slot().index()] = false

  assert_eq(try! owner.select_rows(), 2)
  assert_eq(owner.decode_slots[0], third.slot().index())
  assert_eq(owner.decode_slots[1], first.slot().index())
}

///|
test "submitted plan backpressure precedes preemption policy mutation" {
  let owner = wb_owner_with_capacity(2, 3)
  let first = owner.admit(wb_request(731UL), 0UL)
  let second = owner.admit(wb_request(732UL), 0UL)
  ignore(wb_next_plan(owner))
  ignore(wb_next_plan(owner))
  let urgent = owner.admit(wb_request(733UL), 0UL)
  owner.step = 4UL
  owner.admitted_steps[urgent.slot().index()] = 0UL
  let waiting_before = owner.waiting_count
  let active_before = owner.active_count
  let outcome = try! owner.build_next()
  assert_true(outcome.is_backpressured())
  assert_eq(owner.waiting_count, waiting_before)
  assert_eq(owner.active_count, active_before)
  assert_true(owner.status(first.request_id()).unwrap().in_flight())
  assert_true(owner.status(second.request_id()).unwrap().in_flight())
}

///|
test "in-flight decode is never a recompute preemption victim" {
  let owner = wb_owner_with_capacity(1, 2)
  let active = owner.admit(wb_request(741UL), 0UL)
  make_decode_active(owner, active, 1, 7)
  owner.in_flight[active.slot().index()] = true
  let urgent = owner.admit(wb_request(742UL), 0UL)
  owner.step = 2UL
  owner.admitted_steps[urgent.slot().index()] = 0UL
  let outcome = try! owner.build_next()
  assert_true(outcome.is_idle())
  assert_eq(owner.status(active.request_id()).unwrap().state(), Decode)
  assert_eq(owner.waiting_request_count(), 1)
  assert_eq(owner.free_page_count(), 7)
}

///|
test "eligible aged prompt is selected without destructive preemption" {
  let owner = wb_owner_with_capacity(2, 2)
  let active = owner.admit(wb_request(751UL), 0UL)
  make_decode_active(owner, active, 1, 7)
  let urgent = owner.admit(wb_request(752UL), 0UL)
  owner.step = 2UL
  owner.admitted_steps[urgent.slot().index()] = 0UL
  let outcome = try! owner.build_next()
  assert_true(outcome.is_submitted())
  assert_eq(owner.status(active.request_id()).unwrap().state(), Decode)
  assert_eq(owner.allocated_block_table_count(), 2)
}

///|
test "page pressure can preempt a non-inflight active prefill below active limit" {
  let owner = wb_owner_with_capacity(2, 2)
  let victim = owner.admit(wb_request(761UL), 0UL)
  make_decode_active(owner, victim, 0, 0)
  let victim_slot = victim.slot().index()
  owner.slot_states[victim_slot] = SLOT_PREFILL
  owner.processed_inputs[victim_slot] = 1
  owner.prefill_cursors[victim_slot] = 1
  owner.admitted_steps[victim_slot] = 1UL
  let urgent = owner.admit(wb_request(762UL), 0UL)
  owner.step = 2UL
  owner.admitted_steps[urgent.slot().index()] = 0UL
  let held = try! @page_allocator.PageIdStorage::new(6)
  for index in 0..<6 {
    try! held.write(index, try! owner.page_allocator.allocate_one())
  }

  let outcome = try! owner.build_next()
  assert_true(outcome.is_preempted())
  assert_eq(owner.status(victim.request_id()).unwrap().state(), Waiting)
  assert_eq(
    owner.status(victim.request_id()).unwrap().processed_input_tokens(),
    1,
  )
  assert_false(owner.recompute_required[victim_slot])
  for index in 0..<6 {
    try! owner.page_allocator.release_active(try! held.read(index))
  }
}

///|
test "preemption threshold is exact and victim order is circular" {
  let owner = wb_owner_with_capacity(2, 2)
  let first = owner.admit(wb_request(771UL), 0UL)
  let second = owner.admit(wb_request(772UL), 0UL)
  make_decode_active(owner, first, 1, 7)
  make_decode_active(owner, second, 1, 8)
  let urgent = owner.admit(wb_request(773UL), 0UL)
  owner.admitted_steps[urgent.slot().index()] = 0UL
  owner.step = 1UL
  assert_false(try! owner.preempt_for_aged_waiter())
  owner.step = 2UL
  assert_eq(
    owner.preemption_victim(urgent.slot().index()),
    first.slot().index(),
  )
  owner.preemption_cursor = second.slot().index()
  assert_eq(
    owner.preemption_victim(urgent.slot().index()),
    second.slot().index(),
  )
}

///|
test "cancel and deadline retire recompute waiting authority without KV" {
  let cancelled_owner = wb_owner_with_capacity(1, 2)
  let cancelled = cancelled_owner.admit(wb_request(781UL), 0UL)
  make_decode_active(cancelled_owner, cancelled, 1, 7)
  cancelled_owner.store_generated_token(cancelled.slot().index(), 0, 7)
  let urgent = cancelled_owner.admit(wb_request(782UL), 0UL)
  cancelled_owner.step = 2UL
  cancelled_owner.admitted_steps[urgent.slot().index()] = 0UL
  assert_true((try! cancelled_owner.build_next()).is_preempted())
  ignore(cancelled_owner.cancel(cancelled))
  assert_true(cancelled_owner.status(cancelled.request_id()) is None)
  assert_false(
    cancelled_owner.request_stop_tokens[cancelled.slot().index()].is_live(),
  )
  assert_false(
    try! cancelled_owner.request_tables.is_occupied(cancelled.slot().index()),
  )

  let expired_owner = wb_owner_with_capacity(1, 2)
  let expired = expired_owner.admit(wb_request(783UL), 0UL)
  make_decode_active(expired_owner, expired, 1, 9)
  expired_owner.store_generated_token(expired.slot().index(), 0, 9)
  let blocker = expired_owner.admit(wb_request(784UL), 0UL)
  expired_owner.step = 2UL
  expired_owner.admitted_steps[blocker.slot().index()] = 0UL
  assert_true((try! expired_owner.build_next()).is_preempted())
  ignore(expired_owner.expire_deadlines(100UL))
  assert_true(expired_owner.status(expired.request_id()) is None)
  assert_false(
    expired_owner.request_stop_tokens[expired.slot().index()].is_live(),
  )
  assert_false(
    try! expired_owner.request_tables.is_occupied(expired.slot().index()),
  )
}

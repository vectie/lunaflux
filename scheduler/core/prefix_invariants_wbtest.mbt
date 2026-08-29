///|
test "prefix hit preserves deadline generation history and visible progress" {
  let owner = wb_prefix_scheduler()
  let (seed, seed_lease) = wb_prefix_request(
    941UL,
    [1, 2, 3, 4, 5],
    ReadWrite,
    b"scope",
  )
  ignore(try! owner.admit(seed, 0UL))
  wb_complete_one_token_request(owner, 7)
  ignore(try! owner.take_generated_token())
  ignore(try! owner.take_terminal_notice())
  try! seed_lease.release()

  let (cached, cached_lease) = wb_prefix_request(
    942UL,
    [1, 2, 3, 4, 9],
    ReadOnly,
    b"scope",
    max_new_tokens=2,
  )
  let handle = try! owner.admit(cached, 0UL)
  let before = owner.status(handle.request_id()).unwrap()
  assert_eq(before.processed_input_tokens(), 0)
  assert_eq(before.output_tokens(), 0)
  assert_eq(before.deadline_at_millis(), 100UL)
  assert_eq(before.handle().request_generation(), handle.request_generation())
  let final_prefill = wb_next_plan(owner)
  assert_eq(try! final_prefill.prefill_row(0).sequence_token_start(), 4)
  ignore(
    try! owner.complete(
      final_prefill,
      wb_final_completion(owner, final_prefill, 7),
    ),
  )
  assert_eq((try! owner.take_generated_token()).token_id(), 7)
  let after = owner.status(handle.request_id()).unwrap()
  assert_eq(after.processed_input_tokens(), 5)
  assert_eq(after.output_tokens(), 1)
  assert_eq(after.deadline_at_millis(), 100UL)
  assert_eq(after.handle().request_generation(), handle.request_generation())
  ignore(try! owner.cancel(handle))
  ignore(try! owner.take_terminal_notice())
  try! cached_lease.release()
}

///|
test "prefix re-probe preserves public progress generation deadline and history" {
  let owner = wb_prefix_scheduler()
  let (seed, seed_lease) = wb_prefix_request(
    951UL,
    [1, 2, 3, 4, 5],
    ReadWrite,
    b"scope",
  )
  ignore(try! owner.admit(seed, 0UL))
  wb_complete_one_token_request(owner, 7)
  ignore(try! owner.take_generated_token())
  ignore(try! owner.take_terminal_notice())
  try! seed_lease.release()

  let (cached, cached_lease) = wb_prefix_request(
    952UL,
    [1, 2, 3, 4, 9],
    ReadOnly,
    b"scope",
    max_new_tokens=3,
  )
  let handle = try! owner.admit(cached, 0UL)
  let final_prefill = wb_next_plan(owner)
  ignore(
    try! owner.complete(
      final_prefill,
      wb_final_completion(owner, final_prefill, 7),
    ),
  )
  ignore(try! owner.take_generated_token())
  let slot = handle.slot().index()
  let generation = handle.request_generation()
  assert_eq(owner.processed_inputs[slot], 5)
  assert_eq(owner.output_counts[slot], 1)
  assert_eq(owner.generated_history[owner.history_index(slot, 0)], 7)
  try! owner.release_active_for_recompute(slot)
  let reprobed = owner.status(handle.request_id()).unwrap()
  assert_eq(reprobed.state(), RecomputeWaiting)
  assert_eq(reprobed.processed_input_tokens(), 5)
  assert_eq(reprobed.output_tokens(), 1)
  assert_eq(reprobed.deadline_at_millis(), 100UL)
  assert_eq(reprobed.handle().request_generation(), generation)
  assert_eq(owner.generated_history[owner.history_index(slot, 0)], 7)
  assert_eq(owner.recompute_processed[slot], 0)
  assert_eq(owner.prefix_candidate_tokens[slot], 4)
  ignore(try! owner.cancel(handle))
  ignore(try! owner.take_terminal_notice())
  try! cached_lease.release()
}

///|
test "prefix-rich waiting order is bounded by absolute age priority" {
  let owner = wb_prefix_scheduler()
  let (seed, seed_lease) = wb_prefix_request(
    921UL,
    [1, 2, 3, 4, 5],
    ReadWrite,
    b"scope",
  )
  ignore(try! owner.admit(seed, 0UL))
  wb_complete_one_token_request(owner, 7)
  ignore(try! owner.take_generated_token())
  ignore(try! owner.take_terminal_notice())
  try! seed_lease.release()

  let (miss, miss_lease) = wb_prefix_request(
    922UL,
    [8, 9, 10, 11, 12],
    ReadOnly,
    b"scope",
  )
  let miss_handle = try! owner.admit(miss, 0UL)
  let (hit, hit_lease) = wb_prefix_request(
    923UL,
    [1, 2, 3, 4, 9],
    ReadOnly,
    b"scope",
  )
  let hit_handle = try! owner.admit(hit, 0UL)
  let affinity = wb_next_plan(owner)
  assert_eq(try! affinity.prefill_row(0).request_id(), hit_handle.request_id())
  let writer = try! owner.completion_writer(affinity)
  let hit_row = try! affinity.prefill_row(0)
  try! writer.append_final_prefill_completed(
    slot=try! hit_row.completion_slot(),
    request_id=try! hit_row.request_id(),
    request_generation=try! hit_row.request_generation(),
    processed_tokens=try! hit_row.tokens().length(),
    token_id=7,
  )
  let miss_row = try! affinity.prefill_row(1)
  try! writer.append_prefill_completed(
    slot=try! miss_row.completion_slot(),
    request_id=try! miss_row.request_id(),
    request_generation=try! miss_row.request_generation(),
    processed_tokens=try! miss_row.tokens().length(),
  )
  ignore(try! owner.complete(affinity, try! writer.submit()))
  ignore(try! owner.take_generated_token())
  ignore(try! owner.take_terminal_notice())
  try! hit_lease.release()
  ignore(try! owner.cancel(miss_handle))
  ignore(try! owner.take_terminal_notice())
  try! miss_lease.release()

  let (aged, aged_lease) = wb_prefix_request(
    924UL,
    [8, 9, 10, 11, 12],
    ReadOnly,
    b"scope",
  )
  let aged_handle = try! owner.admit(aged, 0UL)
  let (rich, rich_lease) = wb_prefix_request(
    925UL,
    [1, 2, 3, 4, 9],
    ReadOnly,
    b"scope",
  )
  let rich_handle = try! owner.admit(rich, 0UL)
  owner.admitted_steps[aged_handle.slot().index()] = 0UL
  owner.step = owner.capacity.waiting_age_threshold_steps().to_uint64()
  let age_plan = wb_next_plan(owner)
  assert_eq(try! age_plan.prefill_row(0).request_id(), aged_handle.request_id())
  ignore(try! owner.complete(age_plan, wb_prefill_completion(owner, age_plan)))
  ignore(try! owner.cancel(aged_handle))
  ignore(try! owner.cancel(rich_handle))
  ignore(try! owner.take_terminal_notice())
  ignore(try! owner.take_terminal_notice())
  try! aged_lease.release()
  try! rich_lease.release()
}

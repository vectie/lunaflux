///|
fn replay_assert_plan(
  left : @worker_protocol.SubmittedSchedulePlan,
  right : @worker_protocol.SubmittedSchedulePlan,
) -> Unit raise {
  assert_eq(try! left.sequence_wire(), try! right.sequence_wire())
  assert_eq(
    try! left.model_plan_generation_wire(),
    try! right.model_plan_generation_wire(),
  )
  assert_eq(try! left.token_budget(), try! right.token_budget())
  assert_eq(try! left.row_count(), try! right.row_count())
  assert_eq(try! left.prefill_count(), try! right.prefill_count())
  assert_eq(try! left.decode_count(), try! right.decode_count())
  assert_eq(try! left.token_count(), try! right.token_count())
  assert_eq(try! left.page_count(), try! right.page_count())
  assert_eq(try! left.capability_count(), try! right.capability_count())
  for index in 0..<(try! left.token_count()) {
    assert_eq(try! left.token_at(index), try! right.token_at(index))
  }
  for index in 0..<(try! left.page_count()) {
    assert_eq(try! left.page_at(index), try! right.page_at(index))
  }
  for index in 0..<(try! left.capability_count()) {
    assert_eq(
      (try! left.capability_at(index)).as_int(),
      (try! right.capability_at(index)).as_int(),
    )
  }
  for index in 0..<(try! left.prefill_count()) {
    let left_row = try! left.prefill_row(index)
    let right_row = try! right.prefill_row(index)
    assert_eq(try! left_row.request_id(), try! right_row.request_id())
    assert_eq(
      try! left_row.request_generation(),
      try! right_row.request_generation(),
    )
    assert_eq(
      try! left_row.sequence_token_start(),
      try! right_row.sequence_token_start(),
    )
    assert_eq(try! left_row.tokens(), try! right_row.tokens())
    assert_eq(try! left_row.page_table(), try! right_row.page_table())
    assert_eq(try! left_row.capabilities(), try! right_row.capabilities())
    assert_eq(
      (try! left_row.completion_slot()).value(),
      (try! right_row.completion_slot()).value(),
    )
    assert_eq(try! left_row.produces_token(), try! right_row.produces_token())
    assert_eq(try! left_row.sample_index(), try! right_row.sample_index())
    assert_eq(
      try! left_row.sampling_seed_value(),
      try! right_row.sampling_seed_value(),
    )
    assert_eq(try! left_row.sampling(), try! right_row.sampling())
  }
  for index in 0..<(try! left.decode_count()) {
    let left_row = try! left.decode_row(index)
    let right_row = try! right.decode_row(index)
    assert_eq(try! left_row.request_id(), try! right_row.request_id())
    assert_eq(
      try! left_row.request_generation(),
      try! right_row.request_generation(),
    )
    assert_eq(
      try! left_row.sequence_token_position(),
      try! right_row.sequence_token_position(),
    )
    assert_eq(try! left_row.input_token(), try! right_row.input_token())
    assert_eq(try! left_row.page_table(), try! right_row.page_table())
    assert_eq(try! left_row.capabilities(), try! right_row.capabilities())
    assert_eq(
      (try! left_row.completion_slot()).value(),
      (try! right_row.completion_slot()).value(),
    )
    assert_eq(try! left_row.sample_index(), try! right_row.sample_index())
    assert_eq(
      try! left_row.sampling_seed_value(),
      try! right_row.sampling_seed_value(),
    )
    assert_eq(try! left_row.sampling(), try! right_row.sampling())
  }
}

///|
fn replay_completion(
  owner : Scheduler,
  plan : @worker_protocol.SubmittedSchedulePlan,
  sampled : Int,
) -> @worker_protocol.SubmittedCompletion {
  let writer = try! owner.completion_writer(plan)
  for index in 0..<(try! plan.prefill_count()) {
    let row = try! plan.prefill_row(index)
    if (try! row.produces_token()) {
      try! writer.append_final_prefill_completed(
        slot=try! row.completion_slot(),
        request_id=try! row.request_id(),
        request_generation=try! row.request_generation(),
        processed_tokens=(try! row.tokens()).length(),
        token_id=sampled,
      )
    } else {
      try! writer.append_prefill_completed(
        slot=try! row.completion_slot(),
        request_id=try! row.request_id(),
        request_generation=try! row.request_generation(),
        processed_tokens=(try! row.tokens()).length(),
      )
    }
  }
  try! writer.submit()
}

///|
fn replay_complete_pair(
  left : Scheduler,
  right : Scheduler,
  left_plan : @worker_protocol.SubmittedSchedulePlan,
  right_plan : @worker_protocol.SubmittedSchedulePlan,
) -> Unit raise {
  let left_completion = replay_completion(left, left_plan, 7)
  let right_completion = replay_completion(right, right_plan, 7)
  assert_eq(
    try! left.complete(left_plan, left_completion),
    try! right.complete(right_plan, right_completion),
  )
}

///|
fn replay_assert_telemetry(left : Scheduler, right : Scheduler) -> Unit raise {
  let left_value = left.luna_telemetry()
  let right_value = right.luna_telemetry()
  assert_eq(left_value.queue_depth, right_value.queue_depth)
  assert_eq(left_value.active_requests, right_value.active_requests)
  assert_eq(left_value.kv_pages_used, right_value.kv_pages_used)
  assert_eq(left_value.kv_pages_free, right_value.kv_pages_free)
  assert_eq(left_value.prefix_lookups, right_value.prefix_lookups)
  assert_eq(left_value.prefix_hits, right_value.prefix_hits)
  assert_eq(left_value.prefix_misses, right_value.prefix_misses)
  assert_eq(left_value.prefix_evictions, right_value.prefix_evictions)
  assert_eq(left_value.prefix_tokens_reused, right_value.prefix_tokens_reused)
  assert_eq(
    left_value.prefix_tokens_computed,
    right_value.prefix_tokens_computed,
  )
  assert_eq(left_value.prefix_publications, right_value.prefix_publications)
  assert_eq(left_value.prefix_entries, right_value.prefix_entries)
  assert_eq(left_value.prefix_pages, right_value.prefix_pages)
}

///|
fn replay_assert_status(
  left : Scheduler,
  right : Scheduler,
  handle : RequestHandle,
) -> Unit raise {
  assert_eq(left.status(handle.request_id()), right.status(handle.request_id()))
}

///|
fn replay_assert_generated(
  left : GeneratedTokenNotice,
  right : GeneratedTokenNotice,
) -> Unit raise {
  assert_eq(left.request_id(), right.request_id())
  assert_eq(left.request_generation(), right.request_generation())
  assert_eq(left.token_id(), right.token_id())
  assert_eq(left.position(), right.position())
  assert_eq(left.publication_sequence(), right.publication_sequence())
}

///|
fn replay_seed_pair(
  left : Scheduler,
  right : Scheduler,
  id : UInt64,
) -> Unit raise {
  let (left_seed, left_lease) = wb_prefix_request(
    id,
    [1, 2, 3, 4, 5],
    ReadWrite,
    b"scope",
  )
  let (right_seed, right_lease) = wb_prefix_request(
    id,
    [1, 2, 3, 4, 5],
    ReadWrite,
    b"scope",
  )
  assert_eq(try! left.admit(left_seed, 0UL), try! right.admit(right_seed, 0UL))
  let left_prefill = wb_next_plan(left)
  let right_prefill = wb_next_plan(right)
  replay_assert_plan(left_prefill, right_prefill)
  replay_complete_pair(left, right, left_prefill, right_prefill)
  let left_final = wb_next_plan(left)
  let right_final = wb_next_plan(right)
  replay_assert_plan(left_final, right_final)
  replay_complete_pair(left, right, left_final, right_final)
  replay_assert_generated(
    try! left.take_generated_token(),
    try! right.take_generated_token(),
  )
  assert_eq(try! left.take_terminal_notice(), try! right.take_terminal_notice())
  try! left_lease.release()
  try! right_lease.release()
  replay_assert_telemetry(left, right)
}

///|
test "twin prefix schedulers replay hit miss stale eviction and aged fairness" {
  let left = wb_prefix_scheduler()
  let right = wb_prefix_scheduler()
  replay_seed_pair(left, right, 9811UL)

  let (left_miss, left_miss_lease) = wb_prefix_request(
    9812UL,
    [8, 9, 10, 11, 12],
    ReadOnly,
    b"scope",
  )
  let (right_miss, right_miss_lease) = wb_prefix_request(
    9812UL,
    [8, 9, 10, 11, 12],
    ReadOnly,
    b"scope",
  )
  let left_miss_handle = try! left.admit(left_miss, 0UL)
  let right_miss_handle = try! right.admit(right_miss, 0UL)
  assert_eq(left_miss_handle, right_miss_handle)
  let (left_hit, left_hit_lease) = wb_prefix_request(
    9813UL,
    [1, 2, 3, 4, 9],
    ReadOnly,
    b"scope",
  )
  let (right_hit, right_hit_lease) = wb_prefix_request(
    9813UL,
    [1, 2, 3, 4, 9],
    ReadOnly,
    b"scope",
  )
  let left_hit_handle = try! left.admit(left_hit, 0UL)
  let right_hit_handle = try! right.admit(right_hit, 0UL)
  assert_eq(left_hit_handle, right_hit_handle)
  replay_assert_status(left, right, left_miss_handle)
  replay_assert_status(left, right, left_hit_handle)
  let left_affinity = wb_next_plan(left)
  let right_affinity = wb_next_plan(right)
  replay_assert_plan(left_affinity, right_affinity)
  assert_eq(
    try! left_affinity.prefill_row(0).request_id(),
    left_hit_handle.request_id(),
  )
  replay_complete_pair(left, right, left_affinity, right_affinity)
  replay_assert_generated(
    try! left.take_generated_token(),
    try! right.take_generated_token(),
  )
  assert_eq(try! left.take_terminal_notice(), try! right.take_terminal_notice())
  assert_eq(
    try! left.cancel(left_miss_handle),
    try! right.cancel(right_miss_handle),
  )
  assert_eq(try! left.take_terminal_notice(), try! right.take_terminal_notice())
  try! left_miss_lease.release()
  try! right_miss_lease.release()
  try! left_hit_lease.release()
  try! right_hit_lease.release()
  replay_assert_telemetry(left, right)

  let (left_stale, left_stale_lease) = wb_prefix_request(
    9814UL,
    [1, 2, 3, 4, 9],
    ReadOnly,
    b"scope",
  )
  let (right_stale, right_stale_lease) = wb_prefix_request(
    9814UL,
    [1, 2, 3, 4, 9],
    ReadOnly,
    b"scope",
  )
  let left_stale_handle = try! left.admit(left_stale, 0UL)
  let right_stale_handle = try! right.admit(right_stale, 0UL)
  assert_eq(left_stale_handle, right_stale_handle)
  replay_assert_status(left, right, left_stale_handle)
  assert_eq(left.evict_one_prefix(), right.evict_one_prefix())
  let left_fallback = wb_next_plan(left)
  let right_fallback = wb_next_plan(right)
  replay_assert_plan(left_fallback, right_fallback)
  assert_eq(try! left_fallback.prefill_row(0).sequence_token_start(), 0)
  replay_complete_pair(left, right, left_fallback, right_fallback)
  replay_assert_status(left, right, left_stale_handle)
  assert_eq(
    try! left.cancel(left_stale_handle),
    try! right.cancel(right_stale_handle),
  )
  assert_eq(try! left.take_terminal_notice(), try! right.take_terminal_notice())
  try! left_stale_lease.release()
  try! right_stale_lease.release()
  replay_assert_telemetry(left, right)

  replay_seed_pair(left, right, 9815UL)
  let (left_aged, left_aged_lease) = wb_prefix_request(
    9816UL,
    [8, 9, 10, 11, 12],
    ReadOnly,
    b"scope",
  )
  let (right_aged, right_aged_lease) = wb_prefix_request(
    9816UL,
    [8, 9, 10, 11, 12],
    ReadOnly,
    b"scope",
  )
  let left_aged_handle = try! left.admit(left_aged, 0UL)
  let right_aged_handle = try! right.admit(right_aged, 0UL)
  assert_eq(left_aged_handle, right_aged_handle)
  let (left_rich, left_rich_lease) = wb_prefix_request(
    9817UL,
    [1, 2, 3, 4, 9],
    ReadOnly,
    b"scope",
  )
  let (right_rich, right_rich_lease) = wb_prefix_request(
    9817UL,
    [1, 2, 3, 4, 9],
    ReadOnly,
    b"scope",
  )
  let left_rich_handle = try! left.admit(left_rich, 0UL)
  let right_rich_handle = try! right.admit(right_rich, 0UL)
  assert_eq(left_rich_handle, right_rich_handle)
  replay_assert_status(left, right, left_aged_handle)
  replay_assert_status(left, right, left_rich_handle)
  left.admitted_steps[left_aged_handle.slot().index()] = 0UL
  right.admitted_steps[right_aged_handle.slot().index()] = 0UL
  left.step = left.capacity.waiting_age_threshold_steps().to_uint64()
  right.step = right.capacity.waiting_age_threshold_steps().to_uint64()
  let left_age_plan = wb_next_plan(left)
  let right_age_plan = wb_next_plan(right)
  replay_assert_plan(left_age_plan, right_age_plan)
  assert_eq(
    try! left_age_plan.prefill_row(0).request_id(),
    left_aged_handle.request_id(),
  )
  replay_complete_pair(left, right, left_age_plan, right_age_plan)
  assert_eq(
    try! left.cancel(left_aged_handle),
    try! right.cancel(right_aged_handle),
  )
  assert_eq(
    try! left.cancel(left_rich_handle),
    try! right.cancel(right_rich_handle),
  )
  assert_eq(try! left.take_terminal_notice(), try! right.take_terminal_notice())
  assert_eq(try! left.take_terminal_notice(), try! right.take_terminal_notice())
  try! left_aged_lease.release()
  try! right_aged_lease.release()
  try! left_rich_lease.release()
  try! right_rich_lease.release()
  replay_assert_telemetry(left, right)
  assert_eq(left.live_request_count(), 0)
  assert_eq(right.live_request_count(), 0)
  assert_eq(left.allocated_block_table_count(), 0)
  assert_eq(right.allocated_block_table_count(), 0)
  assert_eq(
    left.prefix_index.unwrap().debug_snapshot(),
    right.prefix_index.unwrap().debug_snapshot(),
  )
  try! left.prefix_index.unwrap().debug_check_invariants()
  try! right.prefix_index.unwrap().debug_check_invariants()
}

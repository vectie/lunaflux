///|
test "resident prompt reaches final demand before opening more prompt folds" {
  for aged in [false, true] {
    let owner = wb_owner_with_capacity(2, 8)
    let first = owner.admit(wb_request(701UL, input_tokens=[1, 2, 3]), 0UL)
    let waiting = owner.admit(wb_request(702UL, input_tokens=[1, 2, 3]), 0UL)
    let initial = wb_next_plan(owner)
    assert_eq(initial.prefill_row(0).request_id(), first.request_id())
    ignore(owner.complete(initial, wb_prefill_completion(owner, initial)))
    assert_eq(owner.processed_inputs[first.slot().index()], 2)
    if aged {
      owner.step = owner.capacity.waiting_age_threshold_steps().to_uint64()
    }
    let next = wb_next_plan(owner)
    assert_eq(next.prefill_row(0).request_id(), first.request_id())
    assert_eq(next.prefill_row(0).sequence_token_start(), 2)
    assert_eq(next.prefill_row(0).tokens().length(), 1)
    assert_eq(next.prefill_row(1).request_id(), waiting.request_id())
    assert_eq(next.prefill_row(1).tokens().length(), 1)
    let writer = owner.completion_writer(next)
    let final_row = next.prefill_row(0)
    writer.append_final_prefill_completed(
      slot=final_row.completion_slot(),
      request_id=final_row.request_id(),
      request_generation=final_row.request_generation(),
      processed_tokens=1,
      token_id=7,
    )
    let partial = next.prefill_row(1)
    writer.append_prefill_completed(
      slot=partial.completion_slot(),
      request_id=partial.request_id(),
      request_generation=partial.request_generation(),
      processed_tokens=1,
    )
    ignore(owner.complete(next, writer.submit()))
    assert_eq(owner.take_generated_token().request_id(), first.request_id())
    assert_eq(owner.processed_inputs[first.slot().index()], 3)
    assert_eq(owner.processed_inputs[waiting.slot().index()], 1)
    assert_true(
      owner.selection_active_visits <= owner.active_slots.length() * 3,
    )
    ignore(owner.cancel(first))
    ignore(owner.cancel(waiting))
    ignore(owner.take_terminal_notice())
    ignore(owner.take_terminal_notice())
    assert_eq(owner.free_page_count(), 8)
  }
}

///|
test "older waiting age overrides resident continuation and in-flight is skipped" {
  let owner = wb_owner_with_capacity(2, 8)
  let waiting = owner.admit(wb_request(711UL), 0UL)
  let active = owner.admit(wb_request(712UL), 0UL)
  make_decode_active(owner, active, 0, 0)
  let slot = active.slot().index()
  owner.slot_states[slot] = SLOT_PREFILL
  owner.processed_inputs[slot] = 1
  owner.prefill_cursors[slot] = 1
  owner.step = 2UL
  assert_eq(owner.oldest_prefill(), waiting.slot().index())
  assert_eq(owner.select_rows(), 2)
  assert_eq(owner.prefill_slots[0], waiting.slot().index())
  // Before the age bound, skip an unavailable resident instead of reserving
  // its tokens or admitting it twice across the paired plan owners.
  owner.step = 0UL
  owner.in_flight[slot] = true
  assert_eq(owner.select_rows(), 2)
  assert_eq(owner.prefill_slots[0], waiting.slot().index())
  assert_eq(owner.prefill_count, 1)
}

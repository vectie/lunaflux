///|
test "waiting unlink touches only the exact intrusive neighbors" {
  let owner = wb_owner_with_capacity(1, 32)
  let handles : Array[RequestHandle] = []
  for id in 1..<=32 {
    handles.push(owner.admit(wb_request(id.to_uint64()), 0UL))
  }
  let removed = handles[15]
  owner.cancel(removed)
  assert_eq(owner.waiting_count, 31)
  assert_eq(owner.waiting_unlink_link_touches, 3)
  assert_eq(
    owner.waiting_next[handles[14].slot().index()],
    handles[16].slot().index(),
  )
  assert_eq(
    owner.waiting_prev[handles[16].slot().index()],
    handles[14].slot().index(),
  )
  assert_eq(owner.waiting_link_generations[removed.slot().index()], 0)
}

///|
test "waiting selection heap is linear to build and preserves prefix FIFO ties" {
  let owner = wb_owner_with_capacity(2, 16)
  let handles : Array[RequestHandle] = []
  for id in 1..<=16 {
    let handle = owner.admit(wb_request(id.to_uint64()), 0UL)
    handles.push(handle)
    owner.prefix_candidate_tokens[handle.slot().index()] = id % 4
  }
  owner.clear_selection()
  owner.prepare_waiting_selection()
  assert_eq(owner.selection_waiting_visits, 16)
  assert_eq(owner.pop_waiting_selection(), handles[2].slot().index())
  assert_eq(owner.pop_waiting_selection(), handles[6].slot().index())
  assert_true(owner.selection_waiting_visits <= owner.waiting_count)
}

///|
test "aged waiting selection gives the exact urgent row absolute priority" {
  let owner = wb_owner_with_capacity(2, 8)
  let first = owner.admit(wb_request(101UL), 0UL)
  ignore(owner.admit(wb_request(102UL), 0UL))
  let third = owner.admit(wb_request(103UL), 0UL)
  owner.prefix_candidate_tokens[third.slot().index()] = 7
  owner.step = 2UL
  owner.admitted_steps[first.slot().index()] = 0UL
  assert_eq(owner.select_rows(), 2)
  assert_eq(owner.prefill_slots[0], first.slot().index())
  assert_eq(owner.prefill_slots[1], NO_INDEX)
  assert_eq(owner.selection_waiting_visits, 0)
}

///|
test "decode selection visits active rows rather than total request capacity" {
  let owner = wb_owner_with_capacity(2, 64)
  let first = owner.admit(wb_request(201UL), 0UL)
  let second = owner.admit(wb_request(202UL), 0UL)
  make_decode_active(owner, first, 1, 7)
  make_decode_active(owner, second, 1, 8)
  for id in 203..<=234 {
    ignore(owner.admit(wb_request(id.to_uint64()), 0UL))
  }
  assert_eq(owner.select_rows(), 2)
  assert_eq(owner.decode_count, 2)
  assert_true(owner.selection_active_visits <= owner.active_count * 3)
  assert_true(owner.selection_active_visits < owner.slot_states.length())
}

///|
test "active swap removal advances a removed decode cursor without starvation" {
  let owner = wb_owner_with_capacity(3, 3)
  let first = owner.admit(wb_request(301UL), 0UL)
  let second = owner.admit(wb_request(302UL), 0UL)
  let third = owner.admit(wb_request(303UL), 0UL)
  make_decode_active(owner, first, 1, 7)
  make_decode_active(owner, second, 1, 8)
  make_decode_active(owner, third, 1, 9)
  owner.decode_cursor = second.slot().index()
  owner.pending_decode_cursor = second.slot().index()
  owner.preemption_cursor = second.slot().index()
  owner.remove_active(second.slot().index())
  assert_eq(owner.active_count, 2)
  assert_eq(owner.decode_cursor, third.slot().index())
  assert_eq(owner.preemption_cursor, third.slot().index())
  assert_eq(owner.active_positions[third.slot().index()], 1)
  assert_eq(owner.select_rows(), 2)
  assert_eq(owner.decode_slots[0], third.slot().index())
  assert_eq(owner.decode_slots[1], first.slot().index())
}

///|
test "active swap removal preserves a moved cursor by slot identity" {
  let owner = wb_owner_with_capacity(3, 3)
  let first = owner.admit(wb_request(311UL), 0UL)
  let second = owner.admit(wb_request(312UL), 0UL)
  let third = owner.admit(wb_request(313UL), 0UL)
  make_decode_active(owner, first, 1, 7)
  make_decode_active(owner, second, 1, 8)
  make_decode_active(owner, third, 1, 9)
  owner.decode_cursor = third.slot().index()
  owner.preemption_cursor = third.slot().index()
  owner.remove_active(first.slot().index())
  assert_eq(owner.active_positions[third.slot().index()], 0)
  assert_eq(owner.decode_cursor, third.slot().index())
  assert_eq(owner.preemption_cursor, third.slot().index())
  assert_eq(owner.select_rows(), 2)
  assert_eq(owner.decode_slots[0], third.slot().index())
  assert_eq(owner.decode_slots[1], second.slot().index())
}

///|
test "deadline heap fast path and mass expiry have bounded structural work" {
  let owner = wb_owner_with_capacity(1, 32)
  for id in 1..<=32 {
    ignore(owner.admit(wb_request(id.to_uint64(), deadline=100UL), 0UL))
  }
  assert_eq(owner.expire_deadlines(99UL), 0)
  assert_eq(owner.deadline_live_visits, 1)
  assert_eq(owner.deadline_due_pops, 0)
  assert_eq(owner.expire_deadlines(100UL), 32)
  assert_eq(owner.deadline_live_visits, 32)
  assert_eq(owner.deadline_due_pops, 32)
  assert_eq(owner.deadline_heap_count, 0)
  assert_true(owner.waiting_unlink_link_touches <= 3)
  for id in 1..<=32 {
    assert_eq(owner.take_terminal_notice().request_id().value(), id.to_uint64())
  }
}

///|
test "deadline publication retains slot order when deadlines differ" {
  let owner = wb_owner_with_capacity(1, 4)
  ignore(owner.admit(wb_request(401UL, deadline=100UL), 0UL))
  ignore(owner.admit(wb_request(402UL, deadline=10UL), 0UL))
  ignore(owner.admit(wb_request(403UL, deadline=50UL), 0UL))
  assert_eq(owner.expire_deadlines(100UL), 3)
  assert_eq(owner.take_terminal_notice().request_id().value(), 401UL)
  assert_eq(owner.take_terminal_notice().request_id().value(), 402UL)
  assert_eq(owner.take_terminal_notice().request_id().value(), 403UL)
}

///|
test "deadline heap identities follow slot generations across reuse" {
  let owner = wb_owner_with_capacity(1, 1)
  let first = owner.admit(wb_request(501UL), 0UL)
  let first_slot = first.slot().index()
  owner.cancel(first)
  ignore(owner.take_terminal_notice())
  let second = owner.admit(wb_request(502UL), 0UL)
  owner.cancel(second)
  ignore(owner.take_terminal_notice())
  let third = owner.admit(wb_request(503UL), 0UL)
  assert_eq(third.slot().index(), first_slot)
  let position = owner.deadline_heap_positions[first_slot]
  assert_eq(
    owner.deadline_heap_generations[position],
    third.slot().generation(),
  )
  assert_true(third.slot().generation() > first.slot().generation())
}

///|
test "panic stale waiting link generation fails closed" {
  let owner = wb_owner_with_capacity(1, 2)
  let first = owner.admit(wb_request(601UL), 0UL)
  owner.waiting_link_generations[first.slot().index()] = 0
  owner.cancel(first)
}

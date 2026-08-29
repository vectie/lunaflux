///|
test "slot generation exhaustion retires capacity without reuse" {
  let owner = wb_owner_with_capacity(1, 1)
  for index in 0..<owner.slot_generations.length() {
    owner.slot_generations[index] = MAX_SLOT_GENERATION - 1
  }
  for request_id in [801UL, 802UL] {
    let handle = owner.admit(wb_request(request_id), 0UL)
    assert_eq(handle.slot().generation(), MAX_SLOT_GENERATION)
    owner.cancel(handle)
  }
  assert_eq(owner.live_request_count(), 0)
  assert_eq(owner.free_head, NO_INDEX)
  assert_eq(owner.free_tail, NO_INDEX)
  for state in owner.slot_states {
    assert_eq(state, SLOT_RETIRED)
  }
  try owner.admit(wb_request(803UL), 0UL) catch {
    Invalid(field=RequestCapacity, issue=Exhausted) => ()
    _ => fail("unexpected retired-capacity admission error")
  } noraise {
    _ => fail("retired scheduler slot was reused")
  }
  for request_id in [801UL, 802UL] {
    let notice = owner.take_terminal_notice()
    assert_eq(notice.request_id(), try! @inference.RequestId::new(request_id))
    assert_eq(notice.reason(), CancelledByCaller)
  }
  assert_true(owner.is_quiescent())
}

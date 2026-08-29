///|
test "Luna endpoint wait takes live stall and clears completed receipt" {
  let coordinator = coordinator_fixture()
  let frame = coordinator_fixture_frame()
  let first = coordinator.offer_luna_framed(frame, source_offset=0, length=1)
  assert_eq(first.kind(), LunaOnlineFramedIngressAccepted)
  assert_eq(first.consumed_bytes(), 1)
  assert_true(coordinator.service.receipt_slot >= 0)
  let receipt_only = coordinator.transport_wait_remaining_millis()
  assert_true(receipt_only > 0)

  let observed = coordinator.service.clock.now_millis()
  coordinator.service.stall_active = true
  coordinator.service.stall_deadline_millis = observed + 100UL
  let combined = coordinator.transport_wait_remaining_millis()
  assert_true(combined > 0)
  assert_true(combined <= 100)
  assert_true(combined <= receipt_only)
  coordinator.service.stall_active = false
  coordinator.service.stall_deadline_millis = 0UL

  let mut cursor = 1
  let mut turns = 0
  while cursor < frame.length() {
    let offered = coordinator.offer_luna_framed(
      frame,
      source_offset=cursor,
      length=frame.length() - cursor,
    )
    let consumed = offered.consumed_bytes()
    if consumed == 0 {
      ignore(coordinator.progress())
    } else {
      cursor += consumed
    }
    turns += 1
    if turns > 4096 {
      fail("Luna endpoint receipt did not complete")
    }
  }
  assert_eq(coordinator.service.receipt_slot, -1)
  assert_eq(coordinator.transport_wait_remaining_millis(), 1000)
  for slot in coordinator.service.work_slots {
    if slot.length() == 1 {
      try! slot[0].discard()
    }
  }
}

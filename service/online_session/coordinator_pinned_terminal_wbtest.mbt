///|
test "Luna queued Ready cannot authorize a read before active retirement" {
  let coordinator = coordinator_fixture()
  let frame = coordinator_fixture_frame()
  let mut cursor = 0
  let mut turns = 0
  while cursor < frame.length() {
    let offered = coordinator.offer_luna_framed(
      frame,
      source_offset=cursor,
      length=frame.length() - cursor,
    )
    let consumed = offered.consumed_bytes()
    if consumed == 0 {
      ignore(coordinator.service.pool.progress())
    } else {
      cursor += consumed
    }
    turns += 1
    if turns > 4096 {
      fail("Luna queued Ready receipt did not converge")
    }
  }
  let work = coordinator.service.queue_head_work()
  let mut state = work.state()
  while state == LunaRequestPreparationPending {
    ignore(coordinator.service.pool.progress())
    state = work.state()
    turns += 1
    if turns > 8192 {
      fail("Luna queued Ready preparation did not converge")
    }
  }
  assert_eq(state, LunaRequestPreparationReady)
  coordinator.service.instance.phase = LunaRequestActivePhase
  coordinator.service.tickets.push({ lane: -1, epoch: 1UL })
  coordinator.service.ticket_sequences.push(1UL)
  coordinator.service.turn_pool = true
  assert_eq(coordinator.progress(), LunaOnlineFramedCoordinatorIdle)
  assert_true(coordinator.stream.luna_framed_input_clear())
  assert_false(coordinator.stream.luna_framed_boundary_clear())
  assert_eq(coordinator.service.queue_count, 1)
  assert_eq(coordinator.service.prepared.length(), 1)
  coordinator.service.tickets.clear()
  coordinator.service.ticket_sequences.clear()
  coordinator.service.prepared[0].discard()
  coordinator.service.prepared.clear()
  coordinator.service.pop_queue_head()
}

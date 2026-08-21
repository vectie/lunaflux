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
      ignore(coordinator.pool.progress())
    } else {
      cursor += consumed
    }
    turns += 1
    if turns > 4096 {
      fail("Luna queued Ready receipt did not converge")
    }
  }
  let work = coordinator.queue_head_work()
  let mut state = work.state()
  while state == LunaRequestPreparationPending {
    ignore(coordinator.pool.progress())
    state = work.state()
    turns += 1
    if turns > 8192 {
      fail("Luna queued Ready preparation did not converge")
    }
  }
  assert_eq(state, LunaRequestPreparationReady)
  coordinator.tickets.push({ epoch: 1UL })
  coordinator.turn_pool = true
  assert_eq(coordinator.progress(), LunaOnlineFramedCoordinatorAdvanced)
  assert_eq(coordinator.queue_count, 1)
  assert_eq(work.state(), LunaRequestPreparationReady)
  coordinator.tickets.clear()
  work.discard()
  coordinator.pop_queue_head()
}

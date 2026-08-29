///|
test "Luna coordinator preserves fragmented receipt awaiting input" {
  let coordinator = coordinator_fixture()
  let frame = coordinator_fixture_frame()
  let ingress = coordinator.offer_luna_framed(frame, source_offset=0, length=1)
  assert_eq(ingress.kind(), LunaOnlineFramedIngressAccepted)
  assert_eq(ingress.consumed_bytes(), 1)
  assert_eq(coordinator.progress(), LunaOnlineFramedCoordinatorAwaitingInput)
  let work = coordinator.service.queue_head_work()
  assert_eq(work.last_work_units(), 0)
  work.cancel()
  ignore(coordinator.service.pool.progress())
  coordinator.service.receipt_slot = -1
  coordinator.service.pop_queue_head()
}

///|
test "Luna AwaitingInput with an active ticket is not a stream boundary" {
  let coordinator = coordinator_fixture()
  let frame = coordinator_fixture_frame()
  coordinator.service.instance.phase = LunaRequestActivePhase
  coordinator.service.tickets.push({ lane: -1, epoch: 1UL })
  coordinator.service.ticket_sequences.push(1UL)
  let ingress = coordinator.offer_luna_framed(frame, source_offset=0, length=1)
  assert_eq(ingress.kind(), LunaOnlineFramedIngressAccepted)
  assert_eq(ingress.consumed_bytes(), 1)
  coordinator.service.turn_pool = true
  assert_eq(coordinator.progress(), LunaOnlineFramedCoordinatorAwaitingInput)
  assert_false(coordinator.stream.luna_framed_boundary_clear())
  let work = coordinator.service.queue_head_work()
  work.cancel()
  ignore(coordinator.service.pool.progress())
  coordinator.service.receipt_slot = -1
  coordinator.service.pop_queue_head()
  coordinator.service.tickets.clear()
  coordinator.service.ticket_sequences.clear()
}

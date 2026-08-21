///|
test "Luna coordinator preserves fragmented receipt awaiting input" {
  let coordinator = coordinator_fixture()
  let frame = coordinator_fixture_frame()
  let ingress = coordinator.offer_luna_framed(frame, source_offset=0, length=1)
  assert_eq(ingress.kind(), LunaOnlineFramedIngressAccepted)
  assert_eq(ingress.consumed_bytes(), 1)
  assert_eq(coordinator.progress(), LunaOnlineFramedCoordinatorAwaitingInput)
  let work = coordinator.queue_head_work()
  assert_eq(work.last_work_units(), 0)
  work.cancel()
  ignore(coordinator.pool.progress())
  coordinator.receipt_slot = -1
  coordinator.pop_queue_head()
}

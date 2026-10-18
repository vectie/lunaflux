///|
test "disconnect consumes a later lane terminal before aborting earlier lanes" {
  let service = coordinator_fixture(multi_lane_count=2).service
  let instance = service.instance
  for index in 0..<2 {
    instance.multi.lanes[index].phase = LunaRequestActivePhase
    instance.multi.lanes[index].generation = 1UL
    service.tickets.push({ lane: index, epoch: 1UL, })
    service.ticket_sequences.push((index + 1).to_uint64())
  }
  instance.multi_publish_usage(1, FailureUsageEvent)
  service.disconnect_requested = true
  service.maintenance_kind = 4
  // No lease or claims are fabricated: the only legal next transition consumes
  // the already-owned event. Trying to abort lane zero first fails this test.
  assert_eq(
    service.progress_maintenance_core(),
    LunaOnlineFramedCoordinatorMaintenanceRequired,
  )
  assert_false(instance.events.has_event())
  assert_eq(instance.multi.retirement_lane, 1)
  assert_true(instance.multi.lanes[1].phase == LunaRequestReleaseReadyPhase)
  assert_true(instance.multi.lanes[0].phase == LunaRequestActivePhase)
  assert_eq(service.maintenance_kind, 2)
}

///|
test "discarded nonterminal events never authorize request release" {
  let instance = coordinator_fixture_instance(multi_lane_count=2)
  let lane = instance.multi.lanes[1]
  lane.phase = LunaRequestAwaitingTerminalPhase
  let effective = @inference.EffectiveLimits::new(
    max_new_tokens=2,
    context_ceiling=8,
    instance.inference,
  )
  instance.events.publish_accepted(instance.expected_model, effective)
  instance.mark_multi_event_published(1, AcceptedEvent)
  assert_eq(
    instance.discard_multi_disconnect_event(),
    LunaOnlineInstancePending,
  )
  assert_true(lane.accepted_acked)
  assert_true(lane.phase == LunaRequestAwaitingTerminalPhase)
  assert_eq(instance.multi.retirement_lane, -1)
  assert_true(instance.event_kind is NoSessionEvent)
  let bytes = FixedArray::make(1, b'x')
  assert_eq(
    instance.events.publish_token_status(
      token_id=1,
      position=0,
      bytes,
      source_offset=0,
      length=1,
    ),
    LunaEventPublishReady,
  )
  instance.mark_multi_event_published(1, TokenEvent)
  assert_eq(
    instance.discard_multi_disconnect_event(),
    LunaOnlineInstancePending,
  )
  assert_true(lane.phase == LunaRequestAwaitingTerminalPhase)
  assert_eq(instance.multi.retirement_lane, -1)
  assert_false(instance.events.has_event())
}

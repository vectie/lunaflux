///|
test "Luna framed offers reject stale substitution and over-confirm" {
  let coordinator = coordinator_fixture()
  let semantic_owner = @luna_event.LunaEventOwner::new(
    coordinator_fixture_inference(),
  )
  semantic_owner.publish_usage(
    input_tokens=2,
    cached_input_tokens=1,
    output_tokens=1,
  )
  let semantic = semantic_owner.view()
  let work = coordinator.service.outbound.begin(semantic)
  while work.state() == LunaFramedEventPending {
    ignore(work.progress())
  }
  let view = work.take_view()
  coordinator.service.outbound_views.push(view)
  coordinator.service.event_length = view.length()
  coordinator.service.event_epoch = 1UL
  assert_true(coordinator.service.begin_stall_deadline())
  let destination = FixedArray::make(64, b'\x00')
  let first = coordinator.copy_framed_event_chunk_to(
    destination,
    destination_offset=0,
    maximum_length=17,
  )
  assert_eq(first.length(), 17)
  try first.confirm(length=0) catch {
    LunaOnlineFramedCoordinatorFailed(LunaOnlineFramedEvent) => ()
    _ => fail("unexpected zero-confirm error")
  } noraise {
    _ => fail("zero Luna event confirmation passed")
  }
  try first.confirm(length=18) catch {
    LunaOnlineFramedCoordinatorFailed(LunaOnlineFramedEvent) => ()
    _ => fail("unexpected over-confirm error")
  } noraise {
    _ => fail("oversized Luna event confirmation passed")
  }
  assert_eq(first.confirm(length=5), LunaOnlineFramedCoordinatorEventReady)
  try first.length() catch {
    LunaOnlineFramedCoordinatorFailed(LunaOnlineFramedEvent) => ()
    _ => fail("unexpected consumed-offer error")
  } noraise {
    _ => fail("partially confirmed Offer retained authority")
  }
  let second = coordinator.copy_framed_event_chunk_to(
    destination,
    destination_offset=0,
    maximum_length=17,
  )
  try first.confirm(length=1) catch {
    LunaOnlineFramedCoordinatorFailed(LunaOnlineFramedEvent) => ()
    _ => fail("unexpected stale Offer substitution error")
  } noraise {
    _ => fail("old Offer confirmed bytes from its successor")
  }
  assert_eq(second.confirm(length=1), LunaOnlineFramedCoordinatorEventReady)
  coordinator.service.outbound_views[0].release()
  coordinator.service.outbound_views.clear()
  coordinator.service.stall_active = false
  coordinator.service.stall_deadline_millis = 0UL
  semantic_owner.retire(semantic)
  semantic_owner.publish_usage(
    input_tokens=2,
    cached_input_tokens=1,
    output_tokens=1,
  )
  let next_semantic = semantic_owner.view()
  let next_work = coordinator.service.outbound.begin(next_semantic)
  while next_work.state() == LunaFramedEventPending {
    ignore(next_work.progress())
  }
  let next_view = next_work.take_view()
  coordinator.service.outbound_views.push(next_view)
  coordinator.service.event_length = next_view.length()
  coordinator.service.event_epoch = 2UL
  coordinator.service.offer_epoch = 0UL
  coordinator.service.confirmed_bytes = 0
  coordinator.service.offered_bytes = 0
  assert_true(coordinator.service.begin_stall_deadline())
  let cross_event = coordinator.copy_framed_event_chunk_to(
    destination,
    destination_offset=0,
    maximum_length=17,
  )
  assert_eq(cross_event.service.offer_epoch, first.service.offer_epoch)
  assert_eq(cross_event.length, first.length)
  try first.confirm(length=1) catch {
    LunaOnlineFramedCoordinatorFailed(LunaOnlineFramedEvent) => ()
    _ => fail("unexpected cross-event stale Offer error")
  } noraise {
    _ => fail("old event Offer confirmed its successor")
  }
  assert_eq(coordinator.service.confirmed_bytes, 0)
  assert_eq(coordinator.service.offered_bytes, cross_event.length)
  assert_eq(
    cross_event.confirm(length=1),
    LunaOnlineFramedCoordinatorEventReady,
  )
  coordinator.service.outbound_views[0].release()
  coordinator.service.outbound_views.clear()
  coordinator.service.stall_active = false
  coordinator.service.stall_deadline_millis = 0UL
  semantic_owner.retire(next_semantic)
}

///|
test "Luna output stall obligation cannot replace an active deadline" {
  let coordinator = coordinator_fixture()
  assert_true(coordinator.service.begin_stall_deadline())
  let deadline = coordinator.service.stall_deadline_millis
  assert_false(coordinator.service.begin_stall_deadline())
  assert_eq(coordinator.service.stall_deadline_millis, deadline)
  assert_true(coordinator.service.stall_active)
  coordinator.service.stall_active = false
  coordinator.service.stall_deadline_millis = 0UL
}

///|
test "Luna coordinator transport wait follows live stall ownership" {
  let coordinator = coordinator_fixture()
  assert_eq(coordinator.transport_wait_remaining_millis(), 1000)
  let observed = coordinator.service.clock.now_millis()
  coordinator.service.stall_active = true
  coordinator.service.stall_deadline_millis = observed + 100UL
  let remaining = coordinator.transport_wait_remaining_millis()
  assert_true(remaining > 0)
  assert_true(remaining <= 100)
  coordinator.service.stall_deadline_millis = 0UL
  assert_eq(coordinator.transport_wait_remaining_millis(), 0)
  coordinator.service.stall_active = false
  coordinator.service.stall_deadline_millis = 0UL
}

///|
test "Luna coordinator rejects a stall wait beyond its envelope" {
  let coordinator = coordinator_fixture()
  let observed = coordinator.service.clock.now_millis()
  coordinator.service.stall_active = true
  coordinator.service.stall_deadline_millis = observed +
    coordinator.service.inference.max_deadline_millis() +
    100UL
  try coordinator.transport_wait_remaining_millis() catch {
    LunaOnlineFramedCoordinatorFailed(LunaOnlineFramedClock) => ()
    _ => fail("unexpected oversized Luna stall wait error")
  } noraise {
    _ => fail("oversized Luna stall wait escaped its inference envelope")
  }
  coordinator.service.stall_active = false
  coordinator.service.stall_deadline_millis = 0UL
}

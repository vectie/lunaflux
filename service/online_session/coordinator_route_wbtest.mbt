///|
test "Luna framed ingress route is stable, opaque, and cancellation exact" {
  let coordinator = coordinator_fixture()
  let frame = coordinator_fixture_frame()
  let first = coordinator.offer_luna_framed(frame, source_offset=0, length=1)
  assert_eq(first.kind(), LunaOnlineFramedIngressAccepted)
  assert_eq(first.consumed_bytes(), 1)
  assert_true(first.has_request_route())
  let route = first.request_route()
  let second = coordinator.offer_luna_framed(frame, source_offset=1, length=1)
  assert_true(second.has_request_route())
  assert_true(route.same_request(second.request_route()))
  assert_false(route.same_request({ sequence: 2UL }))
  coordinator.stream.request_cancel_route(route) catch {
    LunaOnlineFramedCoordinatorFailed(_) => fail("routed cancellation failed")
  }
  let mut turns = 0
  while coordinator.service.queue_count > 0 {
    ignore(
      coordinator.progress() catch {
        LunaOnlineFramedCoordinatorFailed(_) =>
          fail("routed cancellation progress failed")
      },
    )
    turns += 1
    if turns > 8192 {
      fail("Luna routed queued cancellation did not converge")
    }
  }
  assert_eq(coordinator.service.rejection_epoch, 0UL)
  assert_false(coordinator.stream.request_route_live(route))
  try coordinator.stream.request_cancel_route(route) catch {
    LunaOnlineFramedCoordinatorFailed(LunaOnlineFramedOwnership) => ()
    _ => fail("unexpected stale route cancellation error")
  } noraise {
    _ => fail("retired route cancelled a later request")
  }
}

///|
test "Luna rejection credit preserves the exact opaque request route" {
  let coordinator = coordinator_fixture()
  coordinator.service.next_sequence = 7UL
  coordinator.service.publish_rejection(7UL, LunaOnlineFramedRejectedCapacity)
  let rejection = coordinator.take_rejection()
  let route = rejection.request_route()
  assert_true(route.same_request({ sequence: 7UL }))
  assert_false(route.same_request({ sequence: 8UL }))
  rejection.ack()
  try rejection.request_route() catch {
    LunaOnlineFramedCoordinatorFailed(LunaOnlineFramedLifecycle) => ()
    _ => fail("unexpected stale rejection route error")
  } noraise {
    _ => fail("acknowledged rejection retained route authority")
  }
}

///|
test "Luna semantic event route stales with its exact event credit" {
  let compatibility = coordinator_fixture()
  let service = compatibility.service
  retire_fixture_stream(compatibility.stream)
  let stream = open_fixture_semantic_stream(service)
  stage_fixture_semantic_credit(service, 1)
  service.event_route_sequence = 19UL
  let event = stream.take_semantic_event()
  assert_true(event.request_route().same_request({ sequence: 19UL }))
  event.delivered()
  clear_fixture_semantic_credit(service)
  service.event_route_sequence = 0UL
  try event.request_route() catch {
    LunaOnlineFramedCoordinatorFailed(LunaOnlineFramedOwnership) => ()
    _ => fail("unexpected stale semantic route error")
  } noraise {
    _ => fail("retired semantic event retained route authority")
  }
  retire_fixture_stream(stream)
}

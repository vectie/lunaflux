///|
fn open_fixture_semantic_stream(
  service : LunaOnlineFramedService,
) -> LunaOnlineFramedStream raise {
  let opened = service.open_semantic_stream()
  assert_eq(opened.kind(), LunaOnlineFramedOpened)
  service.take_open_stream(opened)
}

///|
fn stage_fixture_semantic_credit(
  service : LunaOnlineFramedService,
  kind : Int,
) -> Unit raise {
  if service.event_epoch == 0xffffffffffffffffUL {
    fail("fixture semantic event epoch exhausted")
  }
  service.event_epoch += 1UL
  service.event_kind = kind
  service.event_route_sequence = service.event_epoch
  service.semantic_event_pending = false
  service.semantic_event_taken = false
  service.event_ack_pending = false
  service.event_credits.push({
    instance: service.instance,
    lane: -1,
    request_epoch: service.event_epoch,
    event_epoch: service.event_epoch,
  })
  assert_true(service.begin_stall_deadline())
}

///|
fn clear_fixture_semantic_credit(service : LunaOnlineFramedService) -> Unit {
  service.event_credits.clear()
  service.event_kind = 0
  service.event_route_sequence = 0UL
  service.semantic_event_pending = false
  service.semantic_event_taken = false
  service.event_ack_pending = false
  service.maintenance_kind = 0
  service.cooperative_handoff = false
  service.stall_active = false
  service.stall_deadline_millis = 0UL
}

///|
test "Luna stream modes do not substitute semantic and framed authority" {
  let compatibility = coordinator_fixture()
  let service = compatibility.service
  let native = compatibility.stream
  try native.take_semantic_event() catch {
    LunaOnlineFramedCoordinatorFailed(LunaOnlineFramedOwnership) => ()
    _ => fail("unexpected native semantic-take error")
  } noraise {
    _ => fail("native stream took semantic delivery authority")
  }
  retire_fixture_stream(native)

  let semantic = open_fixture_semantic_stream(service)
  let destination = FixedArray::make(64, b'\x00')
  try
    semantic.copy_framed_event_chunk_to(
      destination,
      destination_offset=0,
      maximum_length=64,
    )
  catch {
    LunaOnlineFramedCoordinatorFailed(LunaOnlineFramedOwnership) => ()
    _ => fail("unexpected semantic framed-copy error")
  } noraise {
    _ => fail("semantic stream exposed framed Offer authority")
  }
  retire_fixture_stream(semantic)
  assert_eq(service.readiness(), LunaOnlineFramedServiceReady)
}

///|
test "Luna semantic delivery is exact once and ACKs only on later progress" {
  let compatibility = coordinator_fixture()
  let service = compatibility.service
  retire_fixture_stream(compatibility.stream)
  let stream = open_fixture_semantic_stream(service)
  stage_fixture_semantic_credit(service, 3)

  let semantic = stream.take_semantic_event()
  // The synthetic lower credit is intentionally invalid, but reaching the
  // mapped event error proves exact stream/event authentication preceded the
  // lower View projection without consuming delivery authority.
  try semantic.view() catch {
    LunaOnlineFramedCoordinatorFailed(LunaOnlineFramedEvent) => ()
    _ => fail("unexpected synthetic semantic View error")
  } noraise {
    _ => fail("synthetic semantic View unexpectedly became live")
  }
  semantic.delivered()
  assert_false(service.semantic_event_taken)
  assert_true(service.event_ack_pending)
  assert_eq(service.event_credits.length(), 1)
  try semantic.delivered() catch {
    LunaOnlineFramedCoordinatorFailed(LunaOnlineFramedOwnership) => ()
    _ => fail("unexpected repeated semantic delivery error")
  } noraise {
    _ => fail("semantic event delivered twice")
  }

  // Usage replacement is deliberately deferred to the cooperative
  // maintenance turn; delivered itself never calls the lower ACK.
  assert_eq(stream.progress(), LunaOnlineFramedCoordinatorMaintenanceRequired)
  assert_eq(service.maintenance_kind, 3)
  assert_eq(service.event_credits.length(), 1)
  clear_fixture_semantic_credit(service)
  retire_fixture_stream(stream)
}

///|
test "Luna semantic event aliases stale across a reused stream epoch" {
  let compatibility = coordinator_fixture()
  let service = compatibility.service
  retire_fixture_stream(compatibility.stream)
  let first = open_fixture_semantic_stream(service)
  stage_fixture_semantic_credit(service, 1)
  let old = first.take_semantic_event()
  clear_fixture_semantic_credit(service)
  retire_fixture_stream(first)

  let second = open_fixture_semantic_stream(service)
  stage_fixture_semantic_credit(service, 1)
  let current = second.take_semantic_event()
  try old.delivered() catch {
    LunaOnlineFramedCoordinatorFailed(LunaOnlineFramedOwnership) => ()
    _ => fail("unexpected stale semantic event error")
  } noraise {
    _ => fail("old semantic event reached a reused stream epoch")
  }
  current.delivered()
  clear_fixture_semantic_credit(service)
  retire_fixture_stream(second)
}

///|
test "Luna semantic disconnect revokes an untaken event disposition" {
  let compatibility = coordinator_fixture()
  let service = compatibility.service
  retire_fixture_stream(compatibility.stream)
  let stream = open_fixture_semantic_stream(service)
  service.semantic_event_pending = true
  stream.disconnect()
  assert_false(service.semantic_event_pending)
  let mut turns = 0
  while service.stream_phase != 0 {
    ignore(stream.progress())
    turns += 1
    if turns > 128 {
      fail("semantic untaken disconnect did not converge")
    }
  }
  assert_eq(service.stream_mode, 0)
  assert_eq(service.readiness(), LunaOnlineFramedServiceReady)
}

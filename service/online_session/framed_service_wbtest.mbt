///|
fn retire_fixture_stream(stream : LunaOnlineFramedStream) -> Unit raise {
  stream.disconnect()
  let mut turns = 0
  while stream.service.stream_phase != 0 {
    ignore(stream.progress())
    turns += 1
    if turns > 128 {
      fail("Luna reusable stream retirement did not converge")
    }
  }
}

///|
test "Luna framed service reuses one owner across exact stream epochs" {
  let compatibility = coordinator_fixture()
  let service = compatibility.service
  let first = compatibility.stream
  let stale = first
  retire_fixture_stream(first)
  assert_eq(service.readiness(), LunaOnlineFramedServiceReady)

  let second_open = service.open_stream()
  assert_eq(second_open.kind(), LunaOnlineFramedOpened)
  let second = service.take_open_stream(second_open)
  assert_eq(service.readiness(), LunaOnlineFramedServiceStreamActive)
  try stale.progress() catch {
    LunaOnlineFramedCoordinatorFailed(LunaOnlineFramedOwnership) => ()
    _ => fail("unexpected stale stream error")
  } noraise {
    _ => fail("first stream alias reached second stream")
  }
  retire_fixture_stream(second)

  let third_open = service.open_stream()
  assert_eq(third_open.kind(), LunaOnlineFramedOpened)
  try service.take_open_stream(second_open) catch {
    LunaOnlineFramedCoordinatorFailed(LunaOnlineFramedOwnership) => ()
    _ => fail("unexpected stale Open error")
  } noraise {
    _ => fail("retained Open token consumed its successor epoch")
  }
  let third = service.take_open_stream(third_open)
  assert_eq(service.stream_epoch, 3UL)
  retire_fixture_stream(third)
  assert_eq(service.pool.free_lane_count(), service.pool.lane_count())
  assert_eq(service.readiness(), LunaOnlineFramedServiceReady)
}

///|
test "Luna reusable disconnect discards partial receipt without draining owner" {
  let compatibility = coordinator_fixture()
  let service = compatibility.service
  let stream = compatibility.stream
  let frame = coordinator_fixture_frame()
  let offered = stream.offer_luna_framed(frame, source_offset=0, length=1)
  assert_eq(offered.consumed_bytes(), 1)
  assert_eq(service.queue_count, 1)
  retire_fixture_stream(stream)
  assert_false(service.drain_requested)
  assert_false(service.closed)
  assert_eq(service.pool.free_lane_count(), service.pool.lane_count())
  assert_eq(service.readiness(), LunaOnlineFramedServiceReady)
}

///|
fn fixture_trusted_receipt(
  service : LunaOnlineFramedService,
) -> @request_admission.LunaRequestReceipt raise {
  let budget = @inference.DeadlineBudget::from_milliseconds(
    1000UL,
    service.inference,
  )
  @request_admission.LunaRequestReceiptWorkspace::new(service.clock, budget).begin()
}

///|
test "Luna stream transfers trusted receipt only with first admission" {
  let compatibility = coordinator_fixture()
  let service = compatibility.service
  let stream = compatibility.stream
  let receipt = fixture_trusted_receipt(service)
  let frame = coordinator_fixture_frame()
  let admitted = stream.offer_luna_framed_with_receipt(
    receipt,
    frame,
    source_offset=0,
    length=1,
  )
  assert_eq(admitted.consumed_bytes(), 1)
  try receipt.remaining_millis() catch {
    Invalid(LunaPreparedLifecycle) => ()
    _ => fail("unexpected consumed trusted receipt error")
  } noraise {
    _ => fail("admitted trusted receipt remained live")
  }
  retire_fixture_stream(stream)

  let opened = service.open_stream()
  let second = service.take_open_stream(opened)
  service.next_sequence += 1UL
  service.publish_rejection(
    service.next_sequence,
    LunaOnlineFramedRejectedCapacity,
  )
  let retained = fixture_trusted_receipt(service)
  let blocked = second.offer_luna_framed_with_receipt(
    retained,
    frame,
    source_offset=0,
    length=1,
  )
  assert_eq(blocked.kind(), LunaOnlineFramedIngressBackpressured)
  assert_true(retained.remaining_millis() > 0)
  retained.abort()
  let rejection = second.take_rejection()
  rejection.ack()
  retire_fixture_stream(second)
}

///|
fn fill_fixture_stream(
  stream : LunaOnlineFramedStream,
  frame : FixedArray[Byte],
) -> Unit raise {
  let mut cursor = 0
  let mut turns = 0
  while cursor < frame.length() {
    let offered = stream.offer_luna_framed(
      frame,
      source_offset=cursor,
      length=frame.length() - cursor,
    )
    if offered.consumed_bytes() == 0 {
      ignore(stream.progress())
    } else {
      cursor += offered.consumed_bytes()
    }
    turns += 1
    if turns > 4096 {
      fail("Luna reusable receipt did not converge")
    }
  }
  assert_true(stream.luna_framed_receipt_complete())
}

///|
test "Luna framed boundary scalar rejects incomplete and queued receipts" {
  let compatibility = coordinator_fixture()
  let stream = compatibility.stream
  let frame = coordinator_fixture_frame()
  assert_true(stream.luna_framed_boundary_clear())
  assert_true(stream.luna_framed_input_clear())
  let partial = stream.offer_luna_framed(frame, source_offset=0, length=1)
  assert_eq(partial.kind(), LunaOnlineFramedIngressAccepted)
  assert_eq(partial.consumed_bytes(), 1)
  assert_false(stream.luna_framed_boundary_clear())
  assert_false(stream.luna_framed_input_clear())
  let mut cursor = 1
  let mut turns = 0
  while cursor < frame.length() {
    let rest = stream.offer_luna_framed(
      frame,
      source_offset=cursor,
      length=frame.length() - cursor,
    )
    if rest.consumed_bytes() == 0 {
      assert_eq(rest.kind(), LunaOnlineFramedIngressBackpressured)
      ignore(stream.progress())
    } else {
      assert_eq(rest.kind(), LunaOnlineFramedIngressAccepted)
      cursor += rest.consumed_bytes()
    }
    if cursor < frame.length() {
      assert_false(stream.luna_framed_boundary_clear())
    }
    turns += 1
    if turns > 4096 {
      fail("Luna partial receipt did not converge")
    }
  }
  assert_false(stream.luna_framed_boundary_clear())
  assert_true(stream.luna_framed_input_clear())
  retire_fixture_stream(stream)
}

///|
test "Luna framed boundary scalar rejects event and maintenance authority" {
  let compatibility = coordinator_fixture()
  let service = compatibility.service
  let stream = compatibility.stream
  assert_true(stream.luna_framed_boundary_clear())
  service.event_kind = 1
  assert_false(stream.luna_framed_boundary_clear())
  service.event_kind = 0
  service.event_length = 1
  assert_false(stream.luna_framed_boundary_clear())
  service.event_length = 0
  service.semantic_event_pending = true
  assert_false(stream.luna_framed_boundary_clear())
  service.semantic_event_pending = false
  service.event_ack_pending = true
  assert_false(stream.luna_framed_boundary_clear())
  service.event_ack_pending = false
  service.maintenance_kind = 1
  assert_false(stream.luna_framed_boundary_clear())
  service.maintenance_kind = 0
  service.rejection_epoch = 1UL
  assert_false(stream.luna_framed_boundary_clear())
  service.rejection_epoch = 0UL
  service.observation_kind = 1
  assert_false(stream.luna_framed_boundary_clear())
  service.observation_kind = 0
  assert_true(stream.luna_framed_boundary_clear())
  retire_fixture_stream(stream)
}

///|
test "Luna reusable disconnect releases Ready and Prepared lanes" {
  let frame = coordinator_fixture_frame()
  for prepared_case in [false, true] {
    let compatibility = coordinator_fixture()
    let service = compatibility.service
    let stream = compatibility.stream
    if prepared_case {
      service.instance.phase = LunaRequestActivePhase
    }
    fill_fixture_stream(stream, frame)
    let mut turns = 0
    let mut state = try! service.queue_head_work().state()
    while service.prepared.is_empty() && state != LunaRequestPreparationReady {
      ignore(stream.progress())
      if service.prepared.is_empty() {
        state = try! service.queue_head_work().state()
      }
      turns += 1
      if turns > 8192 {
        fail("Luna reusable preparation did not become Ready")
      }
    }
    if prepared_case {
      while service.prepared.is_empty() {
        ignore(stream.progress())
        turns += 1
        if turns > 16384 {
          fail("Luna reusable preparation did not retain Prepared")
        }
      }
    }
    retire_fixture_stream(stream)
    assert_eq(service.pool.free_lane_count(), service.pool.lane_count())
    assert_eq(service.readiness(), LunaOnlineFramedServiceReady)
  }
}

///|
test "Luna old Offer and rejection cannot mutate a reused stream epoch" {
  let compatibility = coordinator_fixture()
  let service = compatibility.service
  let first = compatibility.stream
  let semantic_owner = @luna_event.LunaEventOwner::new(
    coordinator_fixture_inference(),
  )
  semantic_owner.publish_usage(
    input_tokens=2,
    cached_input_tokens=0,
    output_tokens=1,
  )
  let semantic = semantic_owner.view()
  let work = service.outbound.begin(semantic)
  while work.state() == LunaFramedEventPending {
    ignore(work.progress())
  }
  let view = work.take_view()
  service.outbound_views.push(view)
  service.event_length = view.length()
  service.event_epoch += 1UL
  assert_true(service.begin_stall_deadline())
  let output = FixedArray::make(64, b'\x00')
  let old_offer = first.copy_framed_event_chunk_to(
    output,
    destination_offset=0,
    maximum_length=17,
  )
  first.disconnect()
  retire_fixture_stream(first)
  semantic_owner.retire(semantic)

  let opened = service.open_stream()
  let second = service.take_open_stream(opened)
  service.next_sequence += 1UL
  service.publish_rejection(
    service.next_sequence,
    LunaOnlineFramedRejectedFrame,
  )
  let current_rejection = second.take_rejection()
  let old_credit : LunaOnlineFramedRejectionCredit = {
    service,
    stream_epoch: first.epoch,
    epoch: current_rejection.epoch,
  }
  try old_offer.confirm(length=1) catch {
    LunaOnlineFramedCoordinatorFailed(LunaOnlineFramedEvent) => ()
    _ => fail("unexpected cross-stream Offer error")
  } noraise {
    _ => fail("old stream Offer mutated its successor")
  }
  try old_credit.ack() catch {
    LunaOnlineFramedCoordinatorFailed(LunaOnlineFramedLifecycle) => ()
    _ => fail("unexpected cross-stream rejection error")
  } noraise {
    _ => fail("old stream rejection mutated its successor")
  }
  assert_eq(current_rejection.sequence(), service.next_sequence)
  current_rejection.ack()
  retire_fixture_stream(second)
}

///|
test "Luna service drain retires active stream before closing lower owner" {
  let compatibility = coordinator_fixture()
  let service = compatibility.service
  let stale = compatibility.stream
  service.instance.phase = LunaInstanceClosedPhase
  service.begin_drain()
  assert_eq(service.readiness(), LunaOnlineFramedServiceDraining)
  let mut turns = 0
  while service.stream_phase != 0 {
    ignore(service.progress())
    turns += 1
    if turns > 128 {
      fail("Luna service drain did not retire active stream")
    }
  }
  assert_true(service.drain_requested)
  assert_true(service.pool.is_drained())
  try stale.progress() catch {
    LunaOnlineFramedCoordinatorFailed(LunaOnlineFramedOwnership) => ()
    _ => fail("unexpected drained stale stream error")
  } noraise {
    _ => fail("drained stream alias remained live")
  }
}

///|
test "Luna service drain revokes an unopened preallocated stream slot" {
  let compatibility = coordinator_fixture()
  let service = compatibility.service
  retire_fixture_stream(compatibility.stream)
  let opened = service.open_stream()
  assert_eq(opened.kind(), LunaOnlineFramedOpened)
  assert_eq(service.progress(), LunaOnlineFramedServiceStreamActive)
  service.instance.phase = LunaInstanceClosedPhase
  service.begin_drain()
  assert_eq(service.stream_slots.length(), 0)
  assert_eq(service.stream_phase, 0)
  try service.take_open_stream(opened) catch {
    LunaOnlineFramedCoordinatorFailed(LunaOnlineFramedOwnership) => ()
    _ => fail("unexpected unopened stream revocation error")
  } noraise {
    _ => fail("drain preserved unopened stream authority")
  }
}

///|
test "Luna coordinator preserves framed tail and opens FIFO lanes" {
  let coordinator = coordinator_fixture()
  let frame = coordinator_fixture_frame()
  let doubled = FixedArray::make(frame.length() * 2, b'\x00')
  for index in 0..<frame.length() {
    doubled[index] = frame[index]
    doubled[frame.length() + index] = frame[index]
  }
  let mut cursor = 0
  let mut turns = 0
  while cursor < frame.length() {
    let offered = coordinator.offer_luna_framed(
      doubled,
      source_offset=cursor,
      length=doubled.length() - cursor,
    ) catch {
      error =>
        match error {
          LunaOnlineFramedCoordinatorFailed(LunaOnlineFramedClock) =>
            fail("Luna receipt clock became unavailable")
          LunaOnlineFramedCoordinatorFailed(LunaOnlineFramedLifecycle) =>
            fail("Luna receipt Work became stale")
          _ => fail("Luna preparation failed during split receipt")
        }
    }
    let consumed = offered.consumed_bytes()
    assert_true(consumed >= 0)
    assert_true(cursor + consumed <= frame.length())
    if consumed == 0 {
      assert_eq(offered.kind(), LunaOnlineFramedIngressBackpressured)
      ignore(
        coordinator.progress() catch {
          _ => fail("Luna coordinator progress failed during split receipt")
        },
      )
    } else {
      assert_eq(offered.kind(), LunaOnlineFramedIngressAccepted)
      cursor += consumed
    }
    turns += 1
    if turns > 4096 {
      fail("first Luna framed receipt did not converge")
    }
  }
  assert_eq(cursor, frame.length())
  assert_eq(coordinator.service.receipt_slot, -1)
  assert_eq(coordinator.service.queue_count, 1)
  assert_eq(
    coordinator.service.queue_sequences[coordinator.service.queue_head],
    1UL,
  )
  let second = coordinator.offer_luna_framed(
    doubled,
    source_offset=cursor,
    length=doubled.length() - cursor,
  )
  assert_eq(second.kind(), LunaOnlineFramedIngressAccepted)
  assert_true(second.consumed_bytes() > 0)
  assert_eq(coordinator.service.queue_count, 2)
  let second_slot = (coordinator.service.queue_head + 1) %
    coordinator.service.work_slots.length()
  assert_eq(coordinator.service.queue_sequences[second_slot], 2UL)
  for slot in coordinator.service.work_slots {
    if slot.length() == 1 {
      try! slot[0].discard()
    }
  }
}

///|
test "Luna coordinator rejection credit is ordered and alias safe" {
  let coordinator = coordinator_fixture()
  coordinator.service.next_sequence = 1UL
  coordinator.service.publish_rejection(1UL, LunaOnlineFramedRejectedFrame)
  assert_eq(coordinator.progress(), LunaOnlineFramedCoordinatorRejectionReady)
  let first = coordinator.take_rejection()
  assert_eq(first.sequence(), 1UL)
  assert_eq(first.rule(), LunaOnlineFramedRejectedFrame)
  first.ack()
  try first.sequence() catch {
    LunaOnlineFramedCoordinatorFailed(LunaOnlineFramedLifecycle) => ()
    _ => fail("unexpected stale Luna rejection error")
  } noraise {
    _ => fail("acknowledged Luna rejection remained live")
  }
  coordinator.service.next_sequence = 2UL
  coordinator.service.publish_rejection(2UL, LunaOnlineFramedRejectedInput)
  let second = coordinator.take_rejection()
  coordinator.service.stall_deadline_millis = 0UL
  let second_epoch = coordinator.service.rejection_epoch
  let second_rule = coordinator.service.rejection_rule
  try first.ack() catch {
    LunaOnlineFramedCoordinatorFailed(LunaOnlineFramedLifecycle) => ()
    _ => fail("unexpected substituted Luna rejection error")
  } noraise {
    _ => fail("old Luna rejection acknowledged the next result")
  }
  assert_eq(coordinator.service.rejection_epoch, second_epoch)
  assert_eq(coordinator.service.rejection_rule, second_rule)
  assert_true(coordinator.service.rejection_taken)
  assert_false(coordinator.service.disconnect_requested)
  coordinator.service.stall_deadline_millis = 0xffffffffffffffffUL
  assert_eq(second.sequence(), 2UL)
  assert_eq(second.rule(), LunaOnlineFramedRejectedInput)
  second.ack()
}

///|
test "Luna invalid magic publishes typed rejection before retained tail replay" {
  let malformed = coordinator_fixture_frame()
  malformed[0] = malformed[0] ^ b'\x01'
  let coordinator = coordinator_fixture()
  let offered = coordinator.offer_luna_framed(
    malformed,
    source_offset=0,
    length=malformed.length(),
  )
  assert_eq(offered.kind(), LunaOnlineFramedIngressAccepted)
  assert_eq(offered.consumed_bytes(), 16)
  assert_true(offered.consumed_bytes() < malformed.length())
  let mut state = try! coordinator.service.queue_head_work().state()
  let mut turns = 0
  while state == LunaRequestPreparationPending {
    assert_eq(coordinator.progress(), LunaOnlineFramedCoordinatorAdvanced)
    state = try! coordinator.service.queue_head_work().state()
    turns += 1
    if turns > 8 {
      fail("Luna invalid magic scanner did not fail")
    }
  }
  assert_eq(state, LunaRequestPreparationFailed)
  assert_eq(coordinator.progress(), LunaOnlineFramedCoordinatorRejectionReady)
  let rejection = coordinator.take_rejection()
  assert_eq(rejection.sequence(), 1UL)
  assert_eq(rejection.rule(), LunaOnlineFramedRejectedFrame)
  rejection.ack()
}

///|
fn assert_rejection_stall_closes_without_ticket(take : Bool) -> Unit raise {
  let coordinator = coordinator_fixture()
  // A closed inner instance supplies the no-ticket shutdown terminal without
  // requiring a worker fixture; the coordinator itself still owns drain.
  coordinator.service.instance.phase = LunaInstanceClosedPhase
  coordinator.service.next_sequence = 1UL
  coordinator.service.publish_rejection(1UL, LunaOnlineFramedRejectedFrame)
  let mut credit : LunaOnlineFramedRejectionCredit? = None
  if take {
    credit = Some(coordinator.take_rejection())
  }
  coordinator.service.stall_deadline_millis = 0UL
  assert_eq(
    coordinator.progress(),
    LunaOnlineFramedCoordinatorMaintenanceRequired,
  )
  assert_eq(coordinator.service.rejection_epoch, 0UL)
  assert_eq(coordinator.service.maintenance_kind, 0)
  assert_true(coordinator.service.drain_requested)
  match credit {
    Some(value) =>
      try value.sequence() catch {
        LunaOnlineFramedCoordinatorFailed(LunaOnlineFramedLifecycle) => ()
        _ => fail("unexpected expired rejection alias error")
      } noraise {
        _ => fail("expired rejection alias remained live")
      }
    None => ()
  }
  assert_eq(
    coordinator.progress_off_reactor_maintenance(),
    LunaOnlineFramedCoordinatorClosed,
  )
  assert_true(coordinator.is_closed())
}

///|
test "Luna rejection stall drains with taken or untaken no-ticket result" {
  assert_rejection_stall_closes_without_ticket(false)
  assert_rejection_stall_closes_without_ticket(true)
}

///|
test "Luna maintenance kinds reject missing ticket without abort" {
  for kind in [1, 2] {
    let coordinator = coordinator_fixture()
    coordinator.service.maintenance_kind = kind
    try coordinator.progress_off_reactor_maintenance() catch {
      LunaOnlineFramedCoordinatorFailed(LunaOnlineFramedLifecycle) => ()
      _ => fail("unexpected missing-ticket maintenance error")
    } noraise {
      _ => fail("missing-ticket maintenance passed")
    }
  }
}

///|
fn assert_drained_coordinator_closes(
  coordinator : LunaOnlineFramedCoordinator,
) -> Unit raise {
  let mut turns = 0
  while coordinator.service.queue_count > 0 ||
        !coordinator.service.pool.is_drained() {
    ignore(coordinator.progress())
    turns += 1
    if turns > 8192 {
      fail("Luna disconnected preparation queue did not drain")
    }
  }
  assert_eq(coordinator.service.prepared.length(), 0)
  assert_eq(coordinator.service.queue_count, 0)
  assert_eq(coordinator.service.pool.free_lane_count(), 2)
  assert_eq(
    coordinator.progress(),
    LunaOnlineFramedCoordinatorMaintenanceRequired,
  )
  assert_eq(
    coordinator.progress_off_reactor_maintenance(),
    LunaOnlineFramedCoordinatorClosed,
  )
}

///|
fn assert_no_ticket_disconnect_closes(
  coordinator : LunaOnlineFramedCoordinator,
) -> Unit raise {
  coordinator.service.instance.phase = LunaInstanceClosedPhase
  coordinator.disconnect()
  assert_drained_coordinator_closes(coordinator)
}

///|
test "Luna no-ticket disconnect drains partial and ready preparation" {
  let frame = coordinator_fixture_frame()
  let partial = coordinator_fixture()
  let first = partial.offer_luna_framed(frame, source_offset=0, length=1)
  assert_eq(first.consumed_bytes(), 1)
  assert_eq(partial.service.queue_count, 1)
  assert_no_ticket_disconnect_closes(partial)

  let ready = coordinator_fixture()
  let mut cursor = 0
  let mut turns = 0
  while cursor < frame.length() {
    let offered = ready.offer_luna_framed(
      frame,
      source_offset=cursor,
      length=frame.length() - cursor,
    )
    let consumed = offered.consumed_bytes()
    if consumed == 0 {
      ignore(ready.service.pool.progress())
    } else {
      cursor += consumed
    }
    turns += 1
    if turns > 4096 {
      fail("Luna ready disconnect receipt did not converge")
    }
  }
  let mut state = try! ready.service.queue_head_work().state()
  while state == LunaRequestPreparationPending {
    ignore(ready.service.pool.progress())
    state = try! ready.service.queue_head_work().state()
    turns += 1
    if turns > 8192 {
      fail("Luna ready disconnect preparation did not converge")
    }
  }
  assert_eq(state, LunaRequestPreparationReady)
  assert_no_ticket_disconnect_closes(ready)
}

///|
test "Luna close-only retirement drains partial and ready preparation" {
  let frame = coordinator_fixture_frame()
  let partial = coordinator_fixture()
  ignore(partial.offer_luna_framed(frame, source_offset=0, length=1))
  partial.service.instance.phase = LunaInstanceClosedPhase
  assert_eq(
    partial.service.finish_ticket_retirement(
      { lane: -1, epoch: 1UL },
      LunaOnlineInstanceClosed,
    ),
    LunaOnlineFramedCoordinatorDraining,
  )
  assert_false(partial.is_closed())
  assert_drained_coordinator_closes(partial)

  let ready = coordinator_fixture()
  let mut cursor = 0
  let mut turns = 0
  while cursor < frame.length() {
    let offered = ready.offer_luna_framed(
      frame,
      source_offset=cursor,
      length=frame.length() - cursor,
    )
    let consumed = offered.consumed_bytes()
    if consumed == 0 {
      ignore(ready.service.pool.progress())
    } else {
      cursor += consumed
    }
    turns += 1
    if turns > 4096 {
      fail("Luna close-only ready receipt did not converge")
    }
  }
  let mut state = try! ready.service.queue_head_work().state()
  while state == LunaRequestPreparationPending {
    ignore(ready.service.pool.progress())
    state = try! ready.service.queue_head_work().state()
    turns += 1
    if turns > 8192 {
      fail("Luna close-only ready preparation did not converge")
    }
  }
  assert_eq(state, LunaRequestPreparationReady)
  ready.service.instance.phase = LunaInstanceClosedPhase
  assert_eq(
    ready.service.finish_ticket_retirement(
      { lane: -1, epoch: 1UL },
      LunaOnlineInstanceClosed,
    ),
    LunaOnlineFramedCoordinatorDraining,
  )
  assert_false(ready.is_closed())
  assert_drained_coordinator_closes(ready)
}

///|
test "Luna queued failure cannot preempt active request stream" {
  let valid = coordinator_fixture_frame()
  let corrupt = FixedArray::make(valid.length(), b'\x00')
  for index in 0..<valid.length() {
    corrupt[index] = valid[index]
  }
  corrupt[corrupt.length() - 1] = corrupt[corrupt.length() - 1] ^ b'\x01'
  let coordinator = coordinator_fixture()
  coordinator.service.next_sequence = 1UL
  let mut cursor = 0
  let mut turns = 0
  while cursor < corrupt.length() {
    let offered = coordinator.offer_luna_framed(
      corrupt,
      source_offset=cursor,
      length=corrupt.length() - cursor,
    )
    let consumed = offered.consumed_bytes()
    if consumed == 0 {
      ignore(coordinator.service.pool.progress())
    } else {
      cursor += consumed
    }
    turns += 1
    if turns > 4096 {
      fail("Luna corrupt queued receipt did not converge")
    }
  }
  let mut state = try! coordinator.service.queue_head_work().state()
  while state == LunaRequestPreparationPending {
    ignore(coordinator.service.pool.progress())
    state = try! coordinator.service.queue_head_work().state()
    turns += 1
    if turns > 8192 {
      fail("Luna corrupt queued preparation did not fail")
    }
  }
  assert_eq(state, LunaRequestPreparationFailed)
  assert_eq(
    coordinator.service.queue_sequences[coordinator.service.queue_head],
    2UL,
  )
  coordinator.service.tickets.push({ lane: -1, epoch: 1UL })
  coordinator.service.ticket_sequences.push(1UL)
  coordinator.service.turn_pool = true
  assert_eq(coordinator.progress(), LunaOnlineFramedCoordinatorAdvanced)
  assert_eq(coordinator.service.rejection_epoch, 0UL)
  assert_eq(coordinator.service.queue_count, 1)
  coordinator.service.tickets.clear()
  coordinator.service.ticket_sequences.clear()
  assert_eq(coordinator.progress(), LunaOnlineFramedCoordinatorRejectionReady)
  let rejection = coordinator.take_rejection()
  assert_eq(rejection.sequence(), 2UL)
  assert_eq(rejection.rule(), LunaOnlineFramedRejectedFrame)
  rejection.ack()
}

///|
test "Luna Busy dispatch retains Prepared before stale Work retry" {
  let coordinator = coordinator_fixture()
  coordinator.service.instance.phase = LunaRequestActivePhase
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
      ignore(coordinator.progress())
    } else {
      cursor += consumed
    }
    turns += 1
    if turns > 4096 {
      fail("Luna Busy fixture receipt did not converge")
    }
  }
  while coordinator.service.prepared.is_empty() {
    ignore(coordinator.progress())
    turns += 1
    if turns > 8192 {
      fail("Luna Busy fixture did not reach dispatch")
    }
  }
  assert_eq(coordinator.service.prepared.length(), 1)
  assert_eq(coordinator.service.queue_count, 1)
  assert_eq(coordinator.progress(), LunaOnlineFramedCoordinatorIdle)
  assert_eq(coordinator.service.prepared.length(), 1)
  assert_eq(coordinator.service.queue_count, 1)
  coordinator.service.instance.phase = LunaInstanceClosedPhase
  assert_eq(
    coordinator.service.finish_ticket_retirement(
      { lane: -1, epoch: 1UL },
      LunaOnlineInstanceClosed,
    ),
    LunaOnlineFramedCoordinatorDraining,
  )
  assert_false(coordinator.is_closed())
  assert_drained_coordinator_closes(coordinator)
}

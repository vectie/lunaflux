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
  assert_eq(coordinator.receipt_slot, -1)
  assert_eq(coordinator.queue_count, 1)
  assert_eq(coordinator.queue_sequences[coordinator.queue_head], 1UL)
  let second = coordinator.offer_luna_framed(
    doubled,
    source_offset=cursor,
    length=doubled.length() - cursor,
  )
  assert_eq(second.kind(), LunaOnlineFramedIngressAccepted)
  assert_true(second.consumed_bytes() > 0)
  assert_eq(coordinator.queue_count, 2)
  let second_slot = (coordinator.queue_head + 1) %
    coordinator.work_slots.length()
  assert_eq(coordinator.queue_sequences[second_slot], 2UL)
  for slot in coordinator.work_slots {
    if slot.length() == 1 {
      try! slot[0].discard()
    }
  }
}

///|
test "Luna coordinator rejection credit is ordered and alias safe" {
  let coordinator = coordinator_fixture()
  coordinator.next_sequence = 1UL
  coordinator.publish_rejection(1UL, LunaOnlineFramedRejectedFrame)
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
  coordinator.next_sequence = 2UL
  coordinator.publish_rejection(2UL, LunaOnlineFramedRejectedInput)
  let second = coordinator.take_rejection()
  coordinator.stall_deadline_millis = 0UL
  let second_epoch = coordinator.rejection_epoch
  let second_rule = coordinator.rejection_rule
  try first.ack() catch {
    LunaOnlineFramedCoordinatorFailed(LunaOnlineFramedLifecycle) => ()
    _ => fail("unexpected substituted Luna rejection error")
  } noraise {
    _ => fail("old Luna rejection acknowledged the next result")
  }
  assert_eq(coordinator.rejection_epoch, second_epoch)
  assert_eq(coordinator.rejection_rule, second_rule)
  assert_true(coordinator.rejection_taken)
  assert_false(coordinator.disconnect_requested)
  coordinator.stall_deadline_millis = 0xffffffffffffffffUL
  assert_eq(second.sequence(), 2UL)
  assert_eq(second.rule(), LunaOnlineFramedRejectedInput)
  second.ack()
}

///|
fn assert_rejection_stall_closes_without_ticket(take : Bool) -> Unit raise {
  let coordinator = coordinator_fixture()
  // A closed inner instance supplies the no-ticket shutdown terminal without
  // requiring a worker fixture; the coordinator itself still owns drain.
  coordinator.instance.phase = LunaInstanceClosedPhase
  coordinator.next_sequence = 1UL
  coordinator.publish_rejection(1UL, LunaOnlineFramedRejectedFrame)
  let mut credit : LunaOnlineFramedRejectionCredit? = None
  if take {
    credit = Some(coordinator.take_rejection())
  }
  coordinator.stall_deadline_millis = 0UL
  assert_eq(
    coordinator.progress(),
    LunaOnlineFramedCoordinatorMaintenanceRequired,
  )
  assert_eq(coordinator.rejection_epoch, 0UL)
  assert_eq(coordinator.maintenance_kind, 0)
  assert_true(coordinator.drain_requested)
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
    coordinator.maintenance_kind = kind
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
  while coordinator.queue_count > 0 || !coordinator.pool.is_drained() {
    ignore(coordinator.progress())
    turns += 1
    if turns > 8192 {
      fail("Luna disconnected preparation queue did not drain")
    }
  }
  assert_eq(coordinator.prepared.length(), 0)
  assert_eq(coordinator.queue_count, 0)
  assert_eq(coordinator.pool.free_lane_count(), 2)
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
  coordinator.instance.phase = LunaInstanceClosedPhase
  coordinator.disconnect()
  assert_drained_coordinator_closes(coordinator)
}

///|
test "Luna no-ticket disconnect drains partial and ready preparation" {
  let frame = coordinator_fixture_frame()
  let partial = coordinator_fixture()
  let first = partial.offer_luna_framed(frame, source_offset=0, length=1)
  assert_eq(first.consumed_bytes(), 1)
  assert_eq(partial.queue_count, 1)
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
      ignore(ready.pool.progress())
    } else {
      cursor += consumed
    }
    turns += 1
    if turns > 4096 {
      fail("Luna ready disconnect receipt did not converge")
    }
  }
  let mut state = try! ready.queue_head_work().state()
  while state == LunaRequestPreparationPending {
    ignore(ready.pool.progress())
    state = try! ready.queue_head_work().state()
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
  partial.instance.phase = LunaInstanceClosedPhase
  assert_eq(
    partial.finish_ticket_retirement(LunaOnlineInstanceClosed),
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
      ignore(ready.pool.progress())
    } else {
      cursor += consumed
    }
    turns += 1
    if turns > 4096 {
      fail("Luna close-only ready receipt did not converge")
    }
  }
  let mut state = try! ready.queue_head_work().state()
  while state == LunaRequestPreparationPending {
    ignore(ready.pool.progress())
    state = try! ready.queue_head_work().state()
    turns += 1
    if turns > 8192 {
      fail("Luna close-only ready preparation did not converge")
    }
  }
  assert_eq(state, LunaRequestPreparationReady)
  ready.instance.phase = LunaInstanceClosedPhase
  assert_eq(
    ready.finish_ticket_retirement(LunaOnlineInstanceClosed),
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
  coordinator.next_sequence = 1UL
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
      ignore(coordinator.pool.progress())
    } else {
      cursor += consumed
    }
    turns += 1
    if turns > 4096 {
      fail("Luna corrupt queued receipt did not converge")
    }
  }
  let mut state = try! coordinator.queue_head_work().state()
  while state == LunaRequestPreparationPending {
    ignore(coordinator.pool.progress())
    state = try! coordinator.queue_head_work().state()
    turns += 1
    if turns > 8192 {
      fail("Luna corrupt queued preparation did not fail")
    }
  }
  assert_eq(state, LunaRequestPreparationFailed)
  assert_eq(coordinator.queue_sequences[coordinator.queue_head], 2UL)
  coordinator.tickets.push({ epoch: 1UL })
  coordinator.turn_pool = true
  assert_eq(coordinator.progress(), LunaOnlineFramedCoordinatorAdvanced)
  assert_eq(coordinator.rejection_epoch, 0UL)
  assert_eq(coordinator.queue_count, 1)
  coordinator.tickets.clear()
  assert_eq(coordinator.progress(), LunaOnlineFramedCoordinatorRejectionReady)
  let rejection = coordinator.take_rejection()
  assert_eq(rejection.sequence(), 2UL)
  assert_eq(rejection.rule(), LunaOnlineFramedRejectedFrame)
  rejection.ack()
}

///|
test "Luna Busy dispatch retains Prepared before stale Work retry" {
  let coordinator = coordinator_fixture()
  coordinator.instance.phase = LunaRequestActivePhase
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
  while coordinator.prepared.is_empty() {
    ignore(coordinator.progress())
    turns += 1
    if turns > 8192 {
      fail("Luna Busy fixture did not reach dispatch")
    }
  }
  assert_eq(coordinator.prepared.length(), 1)
  assert_eq(coordinator.queue_count, 1)
  assert_eq(coordinator.progress(), LunaOnlineFramedCoordinatorIdle)
  assert_eq(coordinator.prepared.length(), 1)
  assert_eq(coordinator.queue_count, 1)
  coordinator.instance.phase = LunaInstanceClosedPhase
  assert_eq(
    coordinator.finish_ticket_retirement(LunaOnlineInstanceClosed),
    LunaOnlineFramedCoordinatorDraining,
  )
  assert_false(coordinator.is_closed())
  assert_drained_coordinator_closes(coordinator)
}

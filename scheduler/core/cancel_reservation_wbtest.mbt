///|
test "waiting cancellation reservation commits exactly once" {
  let owner = wb_owner()
  let handle = owner.admit(wb_request(701UL), 0UL)
  assert_eq(owner.reserve_cancel_exact(handle), CancelReserved)
  assert_eq(owner.live_request_count(), 1)
  assert_eq(owner.terminal_notice_count(), 0)
  let committed = owner.commit_cancel_reservation()
  let generation = match committed {
    CancelCommitted(value) => value
    _ => fail("expected committed reservation")
  }
  assert_true(generation != handle.request_generation())
  assert_eq(owner.live_request_count(), 0)
  let terminal = owner.take_terminal_notice()
  assert_eq(terminal.request_id(), handle.request_id())
  assert_eq(terminal.request_generation(), generation)
  assert_eq(terminal.reason(), CancelledByCaller)
  assert_eq(owner.commit_cancel_reservation(), CancelReservationAbsent)
  owner.abort_cancel_reservation()
}

///|
fn cancel_reservation_allocation_positive_control() -> Bytes {
  Bytes::make(3, b'R')
}

///|
test "cancellation allocation gate positive control remains reachable" {
  assert_eq(cancel_reservation_allocation_positive_control().length(), 3)
}

///|
test "reservation preflight backpressure and exhaustion mutate nothing" {
  let owner = wb_owner()
  let handle = owner.admit(wb_request(702UL), 0UL)
  owner.terminal_count = owner.terminal_request_ids.length()
  try owner.reserve_cancel_exact(handle) catch {
    Invalid(field=TerminalNoticeCapacity, issue=Backpressured) => ()
    _ => fail("unexpected reservation backpressure")
  } noraise {
    _ => fail("full terminal ring reserved cancellation")
  }
  assert_false(owner.cancel_reserved)
  assert_eq(owner.status(handle.request_id()).unwrap().handle(), handle)
  owner.terminal_count = 0
  owner.generation_cursor = MAX_GENERATION
  try owner.reserve_cancel_exact(handle) catch {
    Invalid(field=RequestGeneration, issue=Exhausted) => ()
    _ => fail("unexpected reservation exhaustion")
  } noraise {
    _ => fail("exhausted generation reserved cancellation")
  }
  assert_false(owner.cancel_reserved)
  assert_eq(owner.live_request_count(), 1)
}

///|
test "stale handle and aliased reservation operations are harmless" {
  let owner = wb_owner()
  let handle = owner.admit(wb_request(710UL), 0UL)
  let stale = {
    slot: handle.slot(),
    request_id: handle.request_id(),
    request_generation: @inference.RequestGeneration::initial(),
  }
  try owner.reserve_cancel_exact(stale) catch {
    Invalid(field=Request, issue=Stale) => ()
    _ => fail("unexpected stale reservation error")
  } noraise {
    _ => fail("stale generation reserved cancellation")
  }
  assert_false(owner.cancel_reserved)
  assert_eq(owner.reserve_cancel_exact(handle), CancelReserved)
  try owner.reserve_cancel_exact(handle) catch {
    Invalid(field=CancellationReservation, issue=InvalidTransition) => ()
    _ => fail("unexpected duplicate reservation error")
  } noraise {
    _ => fail("duplicate reservation succeeded")
  }
  owner.abort_cancel_reservation()
  owner.abort_cancel_reservation()
  assert_eq(owner.status(handle.request_id()).unwrap().handle(), handle)
}

///|
test "reserved dequeue is exact and abort preserves live request" {
  let owner = wb_owner()
  let handle = owner.admit(wb_request(703UL), 0UL)
  let plan = wb_next_plan(owner)
  ignore(owner.complete(plan, wb_final_completion(owner, plan, 10)))
  let status = owner.status(handle.request_id()).unwrap()
  assert_eq(status.state(), Decode)
  assert_eq(owner.reserve_cancel_exact(status.handle()), CancelReserved)
  let notice = owner.take_reserved_generated_token()
  assert_eq(notice.request_id(), handle.request_id())
  assert_eq(notice.request_generation(), handle.request_generation())
  assert_eq(notice.token_id(), 10)
  assert_eq(notice.position(), 0)
  owner.abort_cancel_reservation()
  assert_eq(owner.live_request_count(), 1)
  assert_eq(owner.status(handle.request_id()).unwrap().state(), Decode)
  assert_eq(owner.generated_token_count(), 0)
  owner.cancel(status.handle())
  assert_eq(owner.live_request_count(), 0)
}

///|
test "active reservation blocks every ordinary mutation until commit" {
  let owner = wb_owner()
  let handle = owner.admit(wb_request(704UL), 0UL)
  assert_eq(owner.reserve_cancel_exact(handle), CancelReserved)
  let expected = Invalid(field=CancellationReservation, issue=InvalidTransition)
  try owner.admit(wb_request(705UL), 0UL) catch {
    error => assert_eq(error, expected)
  } noraise {
    _ => fail("reservation allowed admission")
  }
  try owner.build_next() catch {
    error => assert_eq(error, expected)
  } noraise {
    _ => fail("reservation allowed plan construction")
  }
  try owner.cancel(handle) catch {
    error => assert_eq(error, expected)
  } noraise {
    _ => fail("reservation allowed ordinary cancellation")
  }
  try owner.expire_deadlines(200UL) catch {
    error => assert_eq(error, expected)
  } noraise {
    _ => fail("reservation allowed deadline mutation")
  }
  try owner.drain_instance_loss() catch {
    error => assert_eq(error, expected)
  } noraise {
    _ => fail("reservation allowed instance-loss drain")
  }
  assert_true(owner.commit_cancel_reservation() is CancelCommitted(_))
}

///|
test "reserved dequeue rejects a foreign global publication" {
  let owner = wb_owner_with_capacity(2, 2)
  let first = owner.admit(wb_request(706UL), 0UL)
  let second = owner.admit(wb_request(707UL), 0UL)
  let plan = wb_next_plan(owner)
  ignore(owner.complete(plan, wb_final_completion(owner, plan, 11)))
  assert_eq(owner.reserve_cancel_exact(second), CancelReserved)
  try owner.take_reserved_generated_token() catch {
    Invalid(field=CancellationReservation, issue=Mismatch) => ()
    _ => fail("unexpected foreign-publication error")
  } noraise {
    _ => fail("foreign publication was dequeued")
  }
  assert_eq(owner.generated_token_count(), 1)
  owner.abort_cancel_reservation()
  assert_eq(owner.take_generated_token().request_id(), first.request_id())
}

///|
test "one reservation dequeues at most one generated notice" {
  let owner = wb_owner()
  let handle = owner.admit(wb_request(709UL), 0UL)
  let plan = wb_next_plan(owner)
  ignore(owner.complete(plan, wb_final_completion(owner, plan, 14)))
  let status = owner.status(handle.request_id()).unwrap()
  let second_index = owner.output_tail
  owner.output_request_ids[second_index] = handle.request_id()
  owner.output_generations[second_index] = handle.request_generation()
  owner.output_token_ids[second_index] = 15
  owner.output_positions[second_index] = 1
  owner.output_sequences[second_index] = owner.publication_cursor + 1UL
  owner.output_tail = (second_index + 1) % owner.output_request_ids.length()
  owner.output_count += 1
  owner.publication_cursor += 1UL
  assert_eq(owner.reserve_cancel_exact(status.handle()), CancelReserved)
  assert_eq(owner.take_reserved_generated_token().token_id(), 14)
  try owner.take_reserved_generated_token() catch {
    Invalid(field=CancellationReservation, issue=InvalidTransition) => ()
    _ => fail("unexpected repeated dequeue error")
  } noraise {
    _ => fail("one reservation dequeued two notices")
  }
  assert_eq(owner.generated_token_count(), 1)
  owner.abort_cancel_reservation()
  assert_eq(owner.take_generated_token().token_id(), 15)
}

///|
test "naturally retired generation is classified without reservation" {
  let owner = wb_owner()
  let handle = owner.admit(wb_request(708UL), 0UL)
  let prefill = wb_next_plan(owner)
  ignore(owner.complete(prefill, wb_final_completion(owner, prefill, 12)))
  ignore(owner.take_generated_token())
  let decode = wb_next_plan(owner)
  let row = try! decode.decode_row(0)
  let writer = try! owner.completion_writer(decode)
  try! writer.append_decode_completed(
    slot=try! row.completion_slot(),
    request_id=try! row.request_id(),
    request_generation=try! row.request_generation(),
    token_id=13,
  )
  ignore(owner.complete(decode, try! writer.submit()))
  assert_true(owner.status(handle.request_id()) is None)
  assert_eq(owner.reserve_cancel_exact(handle), CancelNaturalTerminal)
  assert_false(owner.cancel_reserved)
  assert_eq(owner.take_generated_token().token_id(), 13)
  assert_eq(owner.take_terminal_notice().reason(), MaximumOutputReached)
}

///|
test "exact expiry preserves an already queued natural terminal" {
  let owner = wb_owner()
  let handle = owner.admit(wb_request(711UL), 0UL)
  let prefill = wb_next_plan(owner)
  ignore(owner.complete(prefill, wb_final_completion(owner, prefill, 16)))
  ignore(owner.take_generated_token())
  let decode = wb_next_plan(owner)
  let row = try! decode.decode_row(0)
  let writer = try! owner.completion_writer(decode)
  try! writer.append_decode_completed(
    slot=try! row.completion_slot(),
    request_id=try! row.request_id(),
    request_generation=try! row.request_generation(),
    token_id=17,
  )
  ignore(owner.complete(decode, try! writer.submit()))
  assert_true(owner.status(handle.request_id()) is None)
  assert_eq(
    owner.expire_exact_or_natural(handle, 1000UL),
    ExactDeadlineNaturalTerminal,
  )
  assert_eq(owner.generated_token_count(), 1)
  assert_eq(owner.terminal_notice_count(), 1)
}

///|
fn wb_final_completion(
  owner : Scheduler,
  plan : @worker_protocol.SubmittedSchedulePlan,
  token : Int,
) -> @worker_protocol.SubmittedCompletion {
  let row = try! plan.prefill_row(0)
  let writer = try! owner.completion_writer(plan)
  try! writer.append_final_prefill_completed(
    slot=try! row.completion_slot(),
    request_id=try! row.request_id(),
    request_generation=try! row.request_generation(),
    processed_tokens=try! row.tokens().length(),
    token_id=token,
  )
  try! writer.submit()
}

///|
test "device-state invalidation fails active requests but preserves waiting work" {
  let owner = wb_owner_with_capacity(1, 2)
  let active = owner.admit(wb_request(499UL), 0UL)
  let waiting = owner.admit(wb_request(500UL), 0UL)
  let plan = wb_next_plan(owner)
  ignore(owner.complete(plan, wb_final_completion(owner, plan, 7)))
  ignore(owner.take_generated_token())
  assert_eq(owner.status(active.request_id()).unwrap().state(), Decode)
  assert_eq(owner.status(waiting.request_id()).unwrap().state(), Waiting)
  assert_eq(owner.free_page_count(), 7)
  assert_eq(owner.allocated_block_table_count(), 1)

  assert_eq(owner.invalidate_device_state(), 1)
  assert_true(owner.status(active.request_id()) is None)
  assert_eq(owner.status(waiting.request_id()).unwrap().state(), Waiting)
  assert_eq(owner.take_terminal_notice().reason(), WorkerFailed)
  assert_eq(owner.free_page_count(), 8)
  assert_eq(owner.allocated_block_table_count(), 0)
}

///|
test "exact invalidation publishes one authenticated failure generation" {
  let owner = wb_owner_with_capacity(1, 1)
  let handle = owner.admit(wb_request(497UL), 0UL)
  let plan = wb_next_plan(owner)
  ignore(owner.complete(plan, wb_final_completion(owner, plan, 7)))
  ignore(owner.take_generated_token())
  ignore(owner.invalidate_device_state_exact(handle))
  let terminal = owner.take_terminal_notice()
  assert_eq(terminal.request_id(), handle.request_id())
  assert_true(terminal.request_generation() != handle.request_generation())
}

///|
test "exact instance-loss drain publishes one authenticated failure generation" {
  let owner = wb_owner_with_capacity(1, 1)
  let handle = owner.admit(wb_request(604UL), 0UL)
  ignore(owner.drain_instance_loss_exact(handle))
  let terminal = owner.take_terminal_notice()
  assert_eq(terminal.request_id(), handle.request_id())
  assert_true(terminal.request_generation() != handle.request_generation())
}

///|
test "exact online recovery rejects more than one live request transactionally" {
  let owner = wb_owner_with_capacity(2, 2)
  let first = owner.admit(wb_request(605UL), 0UL)
  let second = owner.admit(wb_request(606UL), 0UL)
  try owner.invalidate_device_state_exact(first) catch {
    Invalid(field=RequestCapacity, issue=Mismatch) => ()
    _ => fail("unexpected multi-request invalidation error")
  } noraise {
    _ => fail("exact invalidation accepted two live requests")
  }
  try owner.drain_instance_loss_exact(first) catch {
    Invalid(field=RequestCapacity, issue=Mismatch) => ()
    _ => fail("unexpected multi-request drain error")
  } noraise {
    _ => fail("exact instance-loss drain accepted two live requests")
  }
  assert_eq(owner.live_request_count(), 2)
  assert_eq(owner.status(first.request_id()).unwrap().state(), Waiting)
  assert_eq(owner.status(second.request_id()).unwrap().state(), Waiting)
  assert_eq(owner.terminal_notice_count(), 0)
}

///|
test "queued exact terminal cannot authenticate while unrelated request is live" {
  let owner = wb_owner_with_capacity(2, 2)
  let terminal = owner.admit(wb_request(607UL), 0UL)
  let survivor = owner.admit(wb_request(608UL), 0UL)
  owner.cancel(terminal)
  terminal.request_generation = owner.terminal_generations[owner.terminal_head]
  assert_eq(owner.live_request_count(), 1)
  try owner.invalidate_device_state_exact(terminal) catch {
    Invalid(field=RequestCapacity, issue=Mismatch) => ()
    _ => fail("unexpected queued-terminal invalidation isolation error")
  } noraise {
    _ => fail("queued terminal invalidated across another live request")
  }
  try owner.drain_instance_loss_exact(terminal) catch {
    Invalid(field=RequestCapacity, issue=Mismatch) => ()
    _ => fail("unexpected queued-terminal isolation error")
  } noraise {
    _ => fail("queued terminal authenticated across another live request")
  }
  assert_eq(owner.live_request_count(), 1)
  assert_eq(owner.status(survivor.request_id()).unwrap().state(), Waiting)
  assert_eq(owner.terminal_notice_count(), 1)
}

///|

///|
test "instance-loss drain releases active then waiting requests one at a time" {
  let owner = wb_owner_with_capacity(1, 2)
  let active = owner.admit(wb_request(601UL), 0UL)
  let waiting = owner.admit(wb_request(602UL), 0UL)
  let plan = wb_next_plan(owner)
  ignore(owner.complete(plan, wb_final_completion(owner, plan, 7)))
  ignore(owner.take_generated_token())
  assert_eq(owner.status(active.request_id()).unwrap().state(), Decode)
  assert_eq(owner.status(waiting.request_id()).unwrap().state(), Waiting)
  assert_eq(owner.drain_instance_loss(), Remaining(1))
  assert_true(owner.status(active.request_id()) is None)
  assert_eq(owner.free_page_count(), 8)
  assert_eq(owner.allocated_block_table_count(), 0)
  assert_eq(owner.drain_instance_loss(), Complete)
  assert_true(owner.status(waiting.request_id()) is None)
  assert_eq(owner.drain_instance_loss(), Complete)
  assert_eq(owner.take_terminal_notice().reason(), WorkerFailed)
  assert_eq(owner.take_terminal_notice().reason(), WorkerFailed)
}

///|
test "instance-loss drain backpressure mutates no request or KV authority" {
  let owner = wb_owner_with_capacity(1, 1)
  let handle = owner.admit(wb_request(603UL), 0UL)
  owner.terminal_count = owner.terminal_request_ids.length()
  try owner.drain_instance_loss() catch {
    Invalid(field=TerminalNoticeCapacity, issue=Backpressured) => ()
    _ => fail("unexpected instance-loss backpressure error")
  } noraise {
    _ => fail("full terminal ring accepted instance-loss retirement")
  }
  assert_eq(owner.live_request_count(), 1)
  assert_eq(owner.waiting_request_count(), 1)
  assert_eq(owner.status(handle.request_id()).unwrap().state(), Waiting)
  owner.terminal_count = 0
  assert_eq(owner.drain_instance_loss(), Complete)
  assert_eq(owner.live_request_count(), 0)
}

///|
test "device-state invalidation is transactional and rejects submitted work" {
  let owner = wb_owner_with_capacity(1, 1)
  let active = owner.admit(wb_request(498UL), 0UL)
  let plan = wb_next_plan(owner)
  try owner.invalidate_device_state() catch {
    Invalid(field=Plan, issue=InvalidTransition) => ()
    _ => fail("unexpected submitted-device-state error")
  } noraise {
    _ => fail("submitted device state was invalidated")
  }
  assert_true(owner.status(active.request_id()).unwrap().in_flight())
  ignore(owner.complete(plan, wb_final_completion(owner, plan, 8)))
  ignore(owner.take_generated_token())

  owner.terminal_count = owner.terminal_request_ids.length()
  try owner.invalidate_device_state() catch {
    Invalid(field=TerminalNoticeCapacity, issue=Backpressured) => ()
    _ => fail("unexpected invalidation backpressure error")
  } noraise {
    _ => fail("full terminal ring accepted device-state invalidation")
  }
  assert_true(owner.status(active.request_id()) is Some(_))
  assert_eq(owner.free_page_count(), 7)
  assert_eq(owner.allocated_block_table_count(), 1)
  owner.terminal_count = 0
  owner.terminal_head = 0
  owner.terminal_tail = 0
  assert_eq(owner.invalidate_device_state(), 1)
}

///|
test "completion backpressure preserves submitted work for exact retry" {
  let owner = wb_owner()
  let handle = owner.admit(wb_request(501UL), 0UL)
  let plan = wb_next_plan(owner)
  let completion = wb_final_completion(owner, plan, 7)
  owner.output_count = owner.output_request_ids.length()
  assert_eq(owner.complete(plan, completion), CompletionBackpressured)
  let status = owner.status(handle.request_id()).unwrap()
  assert_true(status.in_flight())
  assert_eq(status.processed_input_tokens(), 0)
  assert_eq(owner.plan_a.state(), Submitted)
  assert_eq(owner.completion_a.state(), CompletionSubmitted)
  assert_eq(owner.free_page_count(), 7)
  owner.output_count = 0
  owner.output_head = 0
  owner.output_tail = 0
  assert_eq(owner.complete_submitted(plan), CompletionCommitted)
  assert_false(owner.status(handle.request_id()).unwrap().in_flight())
  assert_eq(owner.take_generated_token().token_id(), 7)
}

///|
test "worker failure completion is synthesized once and reused on retry" {
  let owner = wb_owner()
  let handle = owner.admit(wb_request(502UL), 0UL)
  let plan = wb_next_plan(owner)
  owner.terminal_count = owner.terminal_request_ids.length()
  try owner.fail_plan(plan, ExecutionFailed) catch {
    Invalid(field=TerminalNoticeCapacity, issue=Backpressured) => ()
    _ => fail("expected worker-failure terminal backpressure")
  } noraise {
    _ => fail("full terminal ring accepted a worker failure")
  }
  let submitted_epoch = owner.completion_a.epoch()
  assert_eq(owner.completion_a.state(), CompletionSubmitted)
  assert_true(owner.status(handle.request_id()).unwrap().in_flight())
  assert_eq(owner.live_request_count(), 1)
  owner.terminal_count = 0
  owner.terminal_head = 0
  owner.terminal_tail = 0
  owner.fail_plan(plan, ExecutionFailed)
  assert_eq(owner.completion_a.epoch(), submitted_epoch + 1UL)
  assert_eq(owner.completion_a.state(), CompletionWritable)
  assert_eq(owner.take_terminal_notice().reason(), WorkerFailed)
  assert_eq(owner.live_request_count(), 0)
  assert_eq(owner.free_page_count(), 8)
}

///|
test "semantic invalid submitted completion is replaced by exact failure" {
  let owner = wb_owner()
  let handle = owner.admit(wb_request(503UL), 0UL)
  let plan = wb_next_plan(owner)
  let row = try! plan.prefill_row(0)
  let writer = try! owner.completion_writer(plan)
  try! writer.append_final_prefill_completed(
    slot=try! row.completion_slot(),
    request_id=try! row.request_id(),
    request_generation=try! row.request_generation(),
    processed_tokens=try! row.tokens().length(),
    token_id=7,
  )
  let invalid = try! writer.submit()
  owner.processed_inputs[owner.plan_a_slots[0]] = 1
  try owner.complete(plan, invalid) catch {
    Invalid(field=Completion, issue=Mismatch) => ()
    _ => fail("unexpected semantic completion error")
  } noraise {
    _ => fail("semantic invalid completion was accepted")
  }
  assert_eq(owner.completion_a.state(), CompletionSubmitted)
  owner.replace_submitted_completion_with_failure(plan, ExecutionFailed)
  assert_eq(owner.take_terminal_notice().reason(), WorkerFailed)
  assert_true(owner.status(handle.request_id()) is None)
  assert_eq(owner.completion_a.state(), CompletionWritable)
}

///|
test "completion retirement is sequence ordered across A and B" {
  let owner = wb_owner_with_capacity(3, 3)
  let first = owner.admit(wb_request(511UL), 0UL)
  let second = owner.admit(wb_request(512UL), 0UL)
  let plan_a = wb_next_plan(owner)
  let plan_b = wb_next_plan(owner)
  let completion_a = wb_final_completion(owner, plan_a, 8)
  let completion_b = wb_final_completion(owner, plan_b, 9)
  try owner.complete(plan_b, completion_b) catch {
    Invalid(field=Completion, issue=OutOfOrder) => ()
    _ => fail("expected ordered retirement rejection")
  } noraise {
    _ => fail("plan B retired before plan A")
  }
  assert_true(owner.status(first.request_id()).unwrap().in_flight())
  assert_true(owner.status(second.request_id()).unwrap().in_flight())
  assert_eq(owner.generated_token_count(), 0)
  ignore(owner.complete(plan_a, completion_a))
  ignore(owner.complete(plan_b, completion_b))
  assert_eq(owner.take_generated_token().request_id(), first.request_id())
  assert_eq(owner.take_generated_token().request_id(), second.request_id())
  assert_eq(owner.last_retired_plan_sequence, 2UL)
}

///|
test "non-inflight active cancellation releases KV immediately" {
  let owner = wb_owner()
  let handle = owner.admit(wb_request(521UL), 0UL)
  let plan = wb_next_plan(owner)
  ignore(owner.complete(plan, wb_final_completion(owner, plan, 10)))
  let active = owner.status(handle.request_id()).unwrap()
  assert_eq(active.state(), Decode)
  assert_false(active.in_flight())
  assert_eq(owner.free_page_count(), 7)
  assert_eq(owner.allocated_block_table_count(), 1)
  owner.cancel(active.handle())
  assert_eq(owner.live_request_count(), 0)
  assert_eq(owner.free_page_count(), 8)
  assert_eq(owner.allocated_block_table_count(), 0)
  assert_eq(owner.take_generated_token().token_id(), 10)
  assert_eq(owner.take_terminal_notice().reason(), CancelledByCaller)
}

///|
test "submitted and completed rows retain active page authority until release" {
  let owner = wb_owner()
  let handle = owner.admit(wb_request(522UL), 0UL)
  let plan = wb_next_plan(owner)
  let page = try! plan.page_at(0)
  try! owner.page_allocator.validate_active(page)
  ignore(owner.complete(plan, wb_final_completion(owner, plan, 10)))
  try! owner.page_allocator.validate_active(page)
  let active = owner.status(handle.request_id()).unwrap()
  assert_eq(active.state(), Decode)
  assert_false(active.in_flight())
  owner.cancel(active.handle())
  try owner.page_allocator.validate_active(page) catch {
    PageNotAllocated(actual) => assert_eq(actual, page)
    _ => fail("unexpected released-page authority failure")
  } noraise {
    _ => fail("released request page retained active authority")
  }
}

///|
test "deadline after submit retires stale work without publishing" {
  let owner = wb_owner()
  let handle = owner.admit(wb_request(531UL, deadline=10UL), 0UL)
  let plan = wb_next_plan(owner)
  let completion = wb_final_completion(owner, plan, 11)
  assert_eq(owner.expire_deadlines(10UL), 1)
  ignore(owner.complete(plan, completion))
  assert_eq(owner.generated_token_count(), 0)
  assert_eq(owner.live_request_count(), 0)
  assert_eq(owner.free_page_count(), 8)
  let terminal = owner.take_terminal_notice()
  assert_eq(terminal.request_id(), handle.request_id())
  assert_eq(terminal.reason(), DeadlineReached)
}

///|
test "exact expiry rejects an already-cancelled in-flight generation" {
  let owner = wb_owner()
  let handle = owner.admit(wb_request(532UL, deadline=10UL), 0UL)
  ignore(wb_next_plan(owner))
  let cancelled_generation = owner.cancel_exact(handle)
  let cancelled = owner.status(handle.request_id()).unwrap()
  assert_eq(cancelled.handle().request_generation(), cancelled_generation)
  try owner.expire_exact(cancelled.handle(), 10UL) catch {
    error =>
      assert_eq(error, Invalid(field=RequestState, issue=InvalidTransition))
  } noraise {
    _ => fail("cancelled in-flight request was expired twice")
  }
  assert_eq(owner.terminal_notice_count(), 1)
}

///|
test "terminal backpressure preserves a maximum-output completion for retry" {
  let owner = wb_owner()
  let handle = owner.admit(wb_request(541UL), 0UL)
  let prefill = wb_next_plan(owner)
  ignore(owner.complete(prefill, wb_final_completion(owner, prefill, 12)))
  let decode = wb_next_plan(owner)
  let row = try! decode.decode_row(0)
  let writer = try! owner.completion_writer(decode)
  try! writer.append_decode_completed(
    slot=try! row.completion_slot(),
    request_id=try! row.request_id(),
    request_generation=try! row.request_generation(),
    token_id=13,
  )
  let completion = try! writer.submit()
  owner.terminal_count = owner.terminal_request_ids.length()
  assert_eq(owner.complete(decode, completion), CompletionBackpressured)
  let status = owner.status(handle.request_id()).unwrap()
  assert_true(status.in_flight())
  assert_eq(status.output_tokens(), 1)
  assert_eq(owner.generated_token_count(), 1)
  owner.terminal_count = 0
  owner.terminal_head = 0
  owner.terminal_tail = 0
  assert_eq(owner.complete_submitted(decode), CompletionCommitted)
  assert_eq(owner.generated_token_count(), 2)
  assert_eq(owner.take_generated_token().token_id(), 12)
  assert_eq(owner.take_generated_token().token_id(), 13)
  assert_eq(owner.take_terminal_notice().reason(), MaximumOutputReached)
  assert_eq(owner.live_request_count(), 0)
  assert_eq(owner.free_page_count(), 8)
}

///|
test "A side is reused for sequence three while B remains submitted" {
  let owner = wb_owner_with_capacity(3, 3)
  ignore(owner.admit(wb_request(551UL), 0UL))
  ignore(owner.admit(wb_request(552UL), 0UL))
  ignore(owner.admit(wb_request(553UL), 0UL))
  let plan_a = wb_next_plan(owner)
  let plan_b = wb_next_plan(owner)
  let completion_b = wb_final_completion(owner, plan_b, 17)
  ignore(owner.complete(plan_a, wb_final_completion(owner, plan_a, 14)))
  assert_eq(owner.take_generated_token().token_id(), 14)
  let plan_three = wb_next_plan(owner)
  assert_eq(try! plan_three.sequence_wire(), 3UL)
  assert_eq(owner.plan_a_sequence, 3UL)
  assert_eq(owner.plan_b_sequence, 2UL)
  assert_eq(try! plan_b.sequence_wire(), 2UL)
  assert_eq(try! plan_three.prefill_count(), 1)
  assert_eq(try! plan_three.decode_count(), 0)
  let writer_three = try! owner.completion_writer(plan_three)
  let prefill = try! plan_three.prefill_row(0)
  try! writer_three.append_final_prefill_completed(
    slot=try! prefill.completion_slot(),
    request_id=try! prefill.request_id(),
    request_generation=try! prefill.request_generation(),
    processed_tokens=try! prefill.tokens().length(),
    token_id=15,
  )
  let completion_three = try! writer_three.submit()
  try owner.complete(plan_three, completion_three) catch {
    Invalid(field=Completion, issue=OutOfOrder) => ()
    _ => fail("expected reused A side to wait for sequence two")
  } noraise {
    _ => fail("sequence three retired before sequence two")
  }
  assert_eq(owner.generated_token_count(), 0)
  assert_eq(owner.plan_a_sequence, 3UL)
  ignore(owner.complete(plan_b, completion_b))
  assert_eq(owner.take_generated_token().token_id(), 17)
  ignore(owner.complete(plan_three, completion_three))
  assert_eq(owner.take_generated_token().token_id(), 15)
  assert_eq(owner.plan_a_sequence, 0UL)
  assert_eq(owner.plan_b_sequence, 0UL)
  assert_eq(owner.plan_a.state(), Writable)
  assert_eq(owner.plan_b.state(), Writable)
}

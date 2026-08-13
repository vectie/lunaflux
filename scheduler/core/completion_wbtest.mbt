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
test "completion backpressure preserves submitted work for exact retry" {
  let owner = wb_owner()
  let handle = owner.admit(wb_request(501UL), 0UL)
  let plan = wb_next_plan(owner)
  let completion = wb_final_completion(owner, plan, 7)
  owner.output_count = owner.output_request_ids.length()
  try owner.complete(plan, completion) catch {
    Invalid(field=OutputCapacity, issue=Backpressured) => ()
    _ => fail("expected output backpressure")
  } noraise {
    _ => fail("full output ring accepted a completion")
  }
  let status = owner.status(handle.request_id()).unwrap()
  assert_true(status.in_flight())
  assert_eq(status.processed_input_tokens(), 0)
  assert_eq(owner.plan_a.state(), Submitted)
  assert_eq(owner.completion_a.state(), CompletionSubmitted)
  assert_eq(owner.free_page_count(), 7)
  owner.output_count = 0
  owner.output_head = 0
  owner.output_tail = 0
  owner.complete(plan, completion)
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
  owner.complete(plan_a, completion_a)
  owner.complete(plan_b, completion_b)
  assert_eq(owner.take_generated_token().request_id(), first.request_id())
  assert_eq(owner.take_generated_token().request_id(), second.request_id())
  assert_eq(owner.last_retired_plan_sequence, 2UL)
}

///|
test "non-inflight active cancellation releases KV immediately" {
  let owner = wb_owner()
  let handle = owner.admit(wb_request(521UL), 0UL)
  let plan = wb_next_plan(owner)
  owner.complete(plan, wb_final_completion(owner, plan, 10))
  let active = owner.status(handle.request_id()).unwrap()
  assert_eq(active.state(), Decode)
  assert_false(active.in_flight())
  assert_eq(owner.free_page_count(), 7)
  assert_eq(owner.allocated_block_table_count(), 1)
  owner.cancel(active.handle())
  assert_eq(owner.live_request_count(), 0)
  assert_eq(owner.free_page_count(), 8)
  assert_eq(owner.allocated_block_table_count(), 0)
  assert_eq(owner.take_terminal_notice().reason(), CancelledByCaller)
}

///|
test "submitted and completed rows retain active page authority until release" {
  let owner = wb_owner()
  let handle = owner.admit(wb_request(522UL), 0UL)
  let plan = wb_next_plan(owner)
  let page = try! plan.page_at(0)
  try! owner.page_allocator.validate_active(page)
  owner.complete(plan, wb_final_completion(owner, plan, 10))
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
  owner.complete(plan, completion)
  assert_eq(owner.generated_token_count(), 0)
  assert_eq(owner.live_request_count(), 0)
  assert_eq(owner.free_page_count(), 8)
  let terminal = owner.take_terminal_notice()
  assert_eq(terminal.request_id(), handle.request_id())
  assert_eq(terminal.reason(), DeadlineReached)
}

///|
test "terminal backpressure preserves a maximum-output completion for retry" {
  let owner = wb_owner()
  let handle = owner.admit(wb_request(541UL), 0UL)
  let prefill = wb_next_plan(owner)
  owner.complete(prefill, wb_final_completion(owner, prefill, 12))
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
  try owner.complete(decode, completion) catch {
    Invalid(field=TerminalNoticeCapacity, issue=Backpressured) => ()
    _ => fail("expected terminal backpressure")
  } noraise {
    _ => fail("full terminal ring accepted a terminal completion")
  }
  let status = owner.status(handle.request_id()).unwrap()
  assert_true(status.in_flight())
  assert_eq(status.output_tokens(), 1)
  assert_eq(owner.generated_token_count(), 1)
  owner.terminal_count = 0
  owner.terminal_head = 0
  owner.terminal_tail = 0
  owner.complete(decode, completion)
  assert_eq(owner.generated_token_count(), 2)
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
  owner.complete(plan_a, wb_final_completion(owner, plan_a, 14))
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
  owner.complete(plan_b, completion_b)
  assert_eq(owner.take_generated_token().token_id(), 17)
  owner.complete(plan_three, completion_three)
  assert_eq(owner.take_generated_token().token_id(), 15)
  assert_eq(owner.plan_a_sequence, 0UL)
  assert_eq(owner.plan_b_sequence, 0UL)
  assert_eq(owner.plan_a.state(), Writable)
  assert_eq(owner.plan_b.state(), Writable)
}

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

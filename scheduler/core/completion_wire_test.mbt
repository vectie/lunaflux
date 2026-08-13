///|
fn final_completion_frame(
  plan : @worker_protocol.SubmittedSchedulePlan,
  token : Int,
) -> @worker_wire.ValidatedCompletionFrame raise {
  let completion_owner = @worker_protocol.CompletionBuffer::new(worker_limits())
  let row = plan.prefill_row(0)
  completion_owner.append_final_prefill_completed(
    slot=row.completion_slot(),
    request_id=row.request_id(),
    request_generation=row.request_generation(),
    processed_tokens=row.tokens().length(),
    token_id=token,
  )
  let frame_owner = @worker_wire.CompletionFrameBuffer::new(
    worker_limits(),
    inference_limits(),
  )
  frame_owner.encode_completion(completion_owner.submit_for(plan))
}

///|
fn decode_completion_frame(
  plan : @worker_protocol.SubmittedSchedulePlan,
  token : Int,
) -> @worker_wire.ValidatedCompletionFrame raise {
  let completion_owner = @worker_protocol.CompletionBuffer::new(worker_limits())
  let row = plan.decode_row(0)
  completion_owner.append_decode_completed(
    slot=row.completion_slot(),
    request_id=row.request_id(),
    request_generation=row.request_generation(),
    token_id=token,
  )
  let frame_owner = @worker_wire.CompletionFrameBuffer::new(
    worker_limits(),
    inference_limits(),
  )
  frame_owner.encode_completion(completion_owner.submit_for(plan))
}

///|
test "wire completion is converted before exact scheduler retirement" {
  let owner = scheduler()
  let handle = owner.admit(request(601UL), 0UL)
  let prefill = next_plan(owner)
  let prefill_completion = owner.accept_completion_frame(
    prefill,
    final_completion_frame(prefill, 7),
  )
  assert_true(owner.status(handle.request_id()).unwrap().in_flight())
  assert_eq(owner.generated_token_count(), 0)
  owner.complete(prefill, prefill_completion)
  assert_eq(owner.take_generated_token().token_id(), 7)

  let decode = next_plan(owner)
  let decode_completion = owner.accept_completion_frame(
    decode,
    decode_completion_frame(decode, 8),
  )
  owner.complete(decode, decode_completion)
  assert_eq(owner.take_generated_token().token_id(), 8)
  assert_eq(owner.take_terminal_notice().reason(), MaximumOutputReached)
  assert_eq(owner.live_request_count(), 0)
  assert_eq(owner.free_page_count(), 8)
}

///|
test "foreign frame rejection leaves paired completion writer available" {
  let owner = scheduler()
  ignore(owner.admit(request(603UL), 0UL))
  let plan_a = next_plan(owner)
  ignore(owner.admit(request(604UL), 0UL))
  let plan_b = next_plan(owner)
  let foreign = final_completion_frame(plan_b, 11)
  try owner.accept_completion_frame(plan_a, foreign) catch {
    WorkerWire(InvalidFrame(CompletionEntry)) => ()
    _ => fail("expected foreign frame rejection")
  } noraise {
    _ => fail("foreign frame accepted")
  }
  let completion_a = owner.accept_completion_frame(
    plan_a,
    final_completion_frame(plan_a, 12),
  )
  owner.complete(plan_a, completion_a)
  let completion_b = owner.accept_completion_frame(plan_b, foreign)
  owner.complete(plan_b, completion_b)
  assert_eq(owner.take_generated_token().token_id(), 12)
  assert_eq(owner.take_generated_token().token_id(), 11)
}

///|
fn wb_wire_limits() -> @worker_protocol.WorkerProtocolLimits {
  try! @worker_protocol.WorkerProtocolLimits::new(
    max_prefill_rows=2,
    max_decode_rows=2,
    max_plan_rows=2,
    max_plan_tokens=2,
    max_plan_pages=8,
    max_capabilities=8,
    max_completion_slots=2,
    max_sequence_tokens=16,
    max_token_id=31,
  )
}

///|
fn wb_wire_inference_limits() -> @inference.InferenceLimits {
  try! @inference.InferenceLimits::new(
    max_text_bytes=64,
    max_input_tokens=8,
    max_new_tokens=4,
    max_context_tokens=16,
    max_token_id=31,
    max_stop_token_ids=4,
    max_stop_strings=2,
    max_stop_string_bytes=8,
    max_trace_bytes=8,
    max_cache_scope_bytes=8,
    max_decoded_delta_bytes=8,
    max_deadline_millis=1000UL,
    max_top_k=32,
    max_temperature=4.0,
  )
}

///|
fn wb_completion_frame(
  plan : @worker_protocol.SubmittedSchedulePlan,
  token : Int,
) -> @worker_wire.ValidatedCompletionFrame raise {
  let completion_owner = @worker_protocol.CompletionBuffer::new(
    wb_wire_limits(),
  )
  let row = plan.prefill_row(0)
  completion_owner.append_final_prefill_completed(
    slot=row.completion_slot(),
    request_id=row.request_id(),
    request_generation=row.request_generation(),
    processed_tokens=row.tokens().length(),
    token_id=token,
  )
  let frame_owner = @worker_wire.CompletionFrameBuffer::new(
    wb_wire_limits(),
    wb_wire_inference_limits(),
  )
  frame_owner.encode_completion(completion_owner.submit_for(plan))
}

///|
test "accepted wire completion survives publication backpressure for retry" {
  let owner = wb_owner()
  let handle = owner.admit(wb_request(605UL), 0UL)
  let plan = wb_next_plan(owner)
  let completion = owner.accept_completion_frame(
    plan,
    wb_completion_frame(plan, 9),
  )
  owner.output_count = owner.output_request_ids.length()
  try owner.complete(plan, completion) catch {
    Invalid(field=OutputCapacity, issue=Backpressured) => ()
    _ => fail("expected output backpressure")
  } noraise {
    _ => fail("full output ring accepted completion")
  }
  assert_true(owner.status(handle.request_id()).unwrap().in_flight())
  assert_eq(owner.completion_a.state(), CompletionSubmitted)
  owner.output_count = 0
  owner.output_head = 0
  owner.output_tail = 0
  owner.complete(plan, completion)
  assert_eq(owner.take_generated_token().token_id(), 9)
}

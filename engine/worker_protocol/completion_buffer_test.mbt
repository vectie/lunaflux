///|
fn submitted_plan_for_completion(
  buffer : @worker_protocol.SchedulePlanBuffer,
) -> @worker_protocol.SubmittedSchedulePlan raise {
  append_mixed_buffer(buffer, allocator())
  submit_mixed_buffer(buffer, @worker_protocol.PlanSequence::first(), None)
}

///|
fn append_mixed_completion(
  buffer : @worker_protocol.CompletionBuffer,
) -> Unit raise {
  buffer.append_prefill_completed(
    slot=slot(0),
    request_id=request_id(7UL),
    request_generation=@inference.RequestGeneration::initial(),
    processed_tokens=2,
  )
  buffer.append_decode_completed(
    slot=slot(1),
    request_id=request_id(8UL),
    request_generation=@inference.RequestGeneration::from_wire(3UL),
    token_id=42,
  )
}

///|
test "reusable completion ingress matches immutable completion semantics" {
  let plan_owner = @worker_protocol.SchedulePlanBuffer::new(protocol_limits())
  let plan = submitted_plan_for_completion(plan_owner)
  let owner = @worker_protocol.CompletionBuffer::new(protocol_limits())
  append_mixed_completion(owner)
  let submitted = owner.submit_for(plan)
  submitted.validate_for(plan)
  assert_eq(submitted.sequence(), plan.sequence())
  assert_eq(submitted.model_plan_generation(), plan.model_plan_generation())
  assert_eq(submitted.entry_count(), 2)
  let prefill = submitted.entry(0)
  assert_eq(prefill.kind(), PrefillCompletion)
  assert_eq(prefill.processed_tokens(), 2)
  let decode = submitted.entry(1)
  assert_eq(decode.kind(), DecodeCompletion)
  assert_eq(decode.token_id(), 42)
  let immutable = @worker_protocol.CompletionRecord::new(
    sequence=plan.sequence(),
    model_plan_generation=plan.model_plan_generation(),
    entries=[
      @worker_protocol.CompletionEntry::prefill_completed(
        slot=slot(0),
        request_id=request_id(7UL),
        request_generation=@inference.RequestGeneration::initial(),
        processed_tokens=2,
        limits=protocol_limits(),
      ),
      @worker_protocol.CompletionEntry::decode_completed(
        slot=slot(1),
        request_id=request_id(8UL),
        request_generation=@inference.RequestGeneration::from_wire(3UL),
        token_id=42,
        limits=protocol_limits(),
      ),
    ],
    limits=protocol_limits(),
  )
  assert_eq(prefill.slot(), immutable.entries()[0].slot())
  assert_eq(decode.request_id(), immutable.entries()[1].request_id())
}

///|
test "completion authentication is separate from current publication" {
  let plan_owner = @worker_protocol.SchedulePlanBuffer::new(protocol_limits())
  let plan = submitted_plan_for_completion(plan_owner)
  let owner = @worker_protocol.CompletionBuffer::new(protocol_limits())
  append_mixed_completion(owner)
  let submitted = owner.submit_for(plan)
  submitted.validate_for(plan)
  let entry = submitted.entry(1)
  assert_true(
    entry.is_current(
      request_id(8UL),
      @inference.RequestGeneration::from_wire(3UL),
    ),
  )
  assert_false(
    entry.is_current(
      request_id(8UL),
      @inference.RequestGeneration::from_wire(4UL),
    ),
  )
}

///|
test "completion lifecycle rejects capacity duplicate stale and wrong owner" {
  let plan_owner = @worker_protocol.SchedulePlanBuffer::new(protocol_limits())
  let plan = submitted_plan_for_completion(plan_owner)
  let left = @worker_protocol.CompletionBuffer::new(protocol_limits())
  let right = @worker_protocol.CompletionBuffer::new(protocol_limits())
  append_mixed_completion(left)
  try
    left.append_failed(
      slot=slot(1),
      request_id=request_id(9UL),
      request_generation=@inference.RequestGeneration::initial(),
      failure=CancelledBeforeExecution,
    )
  catch {
    DuplicateCompletionSlot(index=2) => ()
    _ => fail("expected duplicate slot")
  } noraise {
    _ => fail("expected append to fail")
  }
  let submitted = left.submit_for(plan)
  try
    left.append_failed(
      slot=slot(2),
      request_id=request_id(9UL),
      request_generation=@inference.RequestGeneration::initial(),
      failure=CancelledBeforeExecution,
    )
  catch {
    InvalidCompletionBufferState(
      expected=CompletionWritable,
      actual=CompletionSubmitted
    ) => ()
    _ => fail("expected submitted mutation rejection")
  } noraise {
    _ => fail("expected append to fail")
  }
  try right.consume(submitted, plan) catch {
    CompletionBufferMismatch => ()
    _ => fail("expected wrong owner")
  } noraise {
    _ => fail("expected consume to fail")
  }
  let entry = submitted.entry(0)
  let consumed = left.consume(submitted, plan)
  left.reset(consumed)
  try entry.processed_tokens() catch {
    StaleCompletionBufferEpoch => ()
    _ => fail("expected stale completion entry")
  } noraise {
    _ => fail("expected stale read to fail")
  }
  assert_eq(left.state(), CompletionWritable)
}

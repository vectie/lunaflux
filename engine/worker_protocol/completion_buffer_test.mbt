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
test "reusable completion ingress exposes exact typed outcomes" {
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
  assert_eq(prefill.slot(), slot(0))
  assert_eq(decode.request_id(), request_id(8UL))
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

///|
test "completion validation rejects stale plan and model identities" {
  let first_plan_owner = @worker_protocol.SchedulePlanBuffer::new(
    protocol_limits(),
  )
  let first_plan = submitted_plan_for_completion(first_plan_owner)
  let completions = @worker_protocol.CompletionBuffer::new(protocol_limits())
  append_mixed_completion(completions)
  let submitted = completions.submit_for(first_plan)

  let later_plan_owner = @worker_protocol.SchedulePlanBuffer::new(
    protocol_limits(),
  )
  append_mixed_buffer(later_plan_owner, allocator())
  let later_plan = submit_mixed_buffer(
    later_plan_owner,
    plan_sequence(2UL),
    Some(plan_sequence(1UL)),
  )
  try submitted.validate_for(later_plan) catch {
    StalePlanSequence => ()
    _ => fail("expected stale plan sequence")
  } noraise {
    _ => fail("stale plan sequence was accepted")
  }

  let newer_model_owner = @worker_protocol.SchedulePlanBuffer::new(
    protocol_limits(),
  )
  append_mixed_buffer(newer_model_owner, allocator())
  let newer_model = newer_model_owner.submit(
    sequence=@worker_protocol.PlanSequence::first(),
    model_plan_generation=model_generation(2UL),
    token_budget=3,
    None,
    model_generation(2UL),
  )
  try submitted.validate_for(newer_model) catch {
    StaleModelPlanGeneration => ()
    _ => fail("expected stale model generation")
  } noraise {
    _ => fail("stale model generation was accepted")
  }
}

///|
test "completion submission rejects wrong kind count and processed total" {
  let plan_owner = @worker_protocol.SchedulePlanBuffer::new(protocol_limits())
  let plan = submitted_plan_for_completion(plan_owner)

  let wrong_kind = @worker_protocol.CompletionBuffer::new(protocol_limits())
  wrong_kind.append_decode_completed(
    slot=slot(0),
    request_id=request_id(7UL),
    request_generation=@inference.RequestGeneration::initial(),
    token_id=3,
  )
  wrong_kind.append_decode_completed(
    slot=slot(1),
    request_id=request_id(8UL),
    request_generation=@inference.RequestGeneration::from_wire(3UL),
    token_id=4,
  )
  try wrong_kind.submit_for(plan) catch {
    CompletionKindMismatch(index=0) => ()
    _ => fail("expected completion kind mismatch")
  } noraise {
    _ => fail("wrong completion kind was accepted")
  }
  assert_eq(wrong_kind.state(), CompletionWritable)

  let wrong_count = @worker_protocol.CompletionBuffer::new(protocol_limits())
  wrong_count.append_prefill_completed(
    slot=slot(0),
    request_id=request_id(7UL),
    request_generation=@inference.RequestGeneration::initial(),
    processed_tokens=2,
  )
  try wrong_count.submit_for(plan) catch {
    CompletionCountMismatch => ()
    _ => fail("expected completion count mismatch")
  } noraise {
    _ => fail("incomplete completion was accepted")
  }

  let wrong_total = @worker_protocol.CompletionBuffer::new(protocol_limits())
  wrong_total.append_prefill_completed(
    slot=slot(0),
    request_id=request_id(7UL),
    request_generation=@inference.RequestGeneration::initial(),
    processed_tokens=1,
  )
  wrong_total.append_decode_completed(
    slot=slot(1),
    request_id=request_id(8UL),
    request_generation=@inference.RequestGeneration::from_wire(3UL),
    token_id=4,
  )
  try wrong_total.submit_for(plan) catch {
    CompletionTokenCountMismatch(index=0) => ()
    _ => fail("expected processed-token mismatch")
  } noraise {
    _ => fail("wrong processed-token total was accepted")
  }
}

///|
test "completion append validates bounded token and processed counts" {
  let owner = @worker_protocol.CompletionBuffer::new(protocol_limits())
  try
    owner.append_final_prefill_completed(
      slot=slot(0),
      request_id=request_id(1UL),
      request_generation=@inference.RequestGeneration::initial(),
      processed_tokens=1,
      token_id=1025,
    )
  catch {
    InvalidCompletionToken(slot=0) => ()
    _ => fail("expected invalid completion token")
  } noraise {
    _ => fail("out-of-range completion token was accepted")
  }
  try
    owner.append_prefill_completed(
      slot=slot(0),
      request_id=request_id(1UL),
      request_generation=@inference.RequestGeneration::initial(),
      processed_tokens=33,
    )
  catch {
    CompletionTokenCountMismatch(index=0) => ()
    _ => fail("expected processed-token bound")
  } noraise {
    _ => fail("out-of-range processed-token count was accepted")
  }
  assert_eq(owner.state(), CompletionWritable)
}

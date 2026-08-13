///|
test "reusable plan buffer enforces writable submitted retired lifecycle" {
  let buffer = @worker_protocol.SchedulePlanBuffer::new(protocol_limits())
  debug_inspect(buffer.state(), content="Writable")
  inspect(buffer.epoch(), content="1")
  append_mixed_buffer(buffer, allocator())
  let submitted = submit_mixed_buffer(
    buffer,
    @worker_protocol.PlanSequence::first(),
    None,
  )
  let prefill_row = submitted.prefill_row(0)
  let decode_row = submitted.decode_row(0)
  debug_inspect(buffer.state(), content="Submitted")
  inspect(submitted.sequence().value(), content="1")
  inspect(submitted.model_plan_generation().value(), content="1")
  inspect(submitted.token_budget(), content="3")
  inspect(submitted.row_count(), content="2")
  inspect(submitted.token_count(), content="3")
  inspect(submitted.page_count(), content="2")
  inspect(submitted.capability_count(), content="2")
  try
    buffer.append_decode(
      request_id=request_id(9UL),
      request_generation=@inference.RequestGeneration::initial(),
      sequence_token_position=0,
      input_token=13,
      pages=[allocator().allocate_one()],
      capabilities=[@plan.KernelCapabilityId::causal_attention()],
      sampling=@inference.SamplingParameters::greedy(),
      sampling_seed=sampling_seed(),
      sample_index=0,
      completion_slot=slot(2),
    )
  catch {
    @worker_protocol.InvalidPlanBufferState(expected=Writable, actual=Submitted) =>
      ()
    _ => fail("expected submitted mutation rejection")
  } noraise {
    _ => fail("expected append to fail")
  }
  let retired = buffer.retire(submitted)
  debug_inspect(buffer.state(), content="Retired")
  try submitted.token_at(0) catch {
    InvalidPlanBufferState(expected=Submitted, actual=Retired) => ()
    _ => fail("expected retired handle rejection")
  } noraise {
    _ => fail("expected read to fail")
  }
  buffer.reset(retired)
  debug_inspect(buffer.state(), content="Writable")
  inspect(buffer.epoch(), content="2")
  try submitted.token_at(0) catch {
    StalePlanBufferEpoch => ()
    _ => fail("expected stale epoch")
  } noraise {
    _ => fail("expected old handle to fail")
  }
  try prefill_row.request_id() catch {
    StalePlanBufferEpoch => ()
    _ => fail("expected stale prefill-row epoch")
  } noraise {
    _ => fail("expected old prefill row to fail")
  }
  try decode_row.sampling() catch {
    StalePlanBufferEpoch => ()
    _ => fail("expected stale decode-row epoch")
  } noraise {
    _ => fail("expected old decode row to fail")
  }
}

///|
test "retirement and reset authenticate the exact owner and epoch" {
  let left = @worker_protocol.SchedulePlanBuffer::new(protocol_limits())
  let right = @worker_protocol.SchedulePlanBuffer::new(protocol_limits())
  append_mixed_buffer(left, allocator())
  append_mixed_buffer(right, allocator())
  let left_plan = submit_mixed_buffer(
    left,
    @worker_protocol.PlanSequence::first(),
    None,
  )
  let right_plan = submit_mixed_buffer(
    right,
    @worker_protocol.PlanSequence::first(),
    None,
  )
  try right.retire(left_plan) catch {
    PlanBufferMismatch => ()
    _ => fail("expected owner mismatch")
  } noraise {
    _ => fail("expected retirement to fail")
  }
  let left_retired = left.retire(left_plan)
  let right_retired = right.retire(right_plan)
  try right.reset(left_retired) catch {
    PlanBufferMismatch => ()
    _ => fail("expected retirement proof mismatch")
  } noraise {
    _ => fail("expected reset to fail")
  }
  left.reset(left_retired)
  right.reset(right_retired)
  try left.reset(left_retired) catch {
    StalePlanBufferEpoch => ()
    _ => fail("expected stale retirement proof")
  } noraise {
    _ => fail("expected repeated reset to fail")
  }
}

///|
test "two independently owned buffers model N and N plus one overlap" {
  let plan_n = @worker_protocol.SchedulePlanBuffer::new(protocol_limits())
  let plan_next = @worker_protocol.SchedulePlanBuffer::new(protocol_limits())
  append_mixed_buffer(plan_n, allocator())
  let submitted_n = submit_mixed_buffer(
    plan_n,
    @worker_protocol.PlanSequence::first(),
    None,
  )
  append_mixed_buffer(plan_next, allocator())
  let submitted_next = submit_mixed_buffer(
    plan_next,
    plan_sequence(2UL),
    Some(plan_sequence(1UL)),
  )
  debug_inspect(
    (
      submitted_n.sequence().value(),
      submitted_next.sequence().value(),
      plan_n.state(),
      plan_next.state(),
    ),
    content="(1, 2, Submitted, Submitted)",
  )
  let retired_n = plan_n.retire(submitted_n)
  plan_n.reset(retired_n)
  debug_inspect(
    (plan_n.state(), plan_next.state()),
    content="(Writable, Submitted)",
  )
  append_mixed_buffer(plan_n, allocator())
  let submitted_after_next = submit_mixed_buffer(
    plan_n,
    plan_sequence(3UL),
    Some(plan_sequence(2UL)),
  )
  inspect(submitted_after_next.sequence().value(), content="3")
}

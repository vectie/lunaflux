///|
fn append_mixed_writer(
  writer : @worker_protocol.CompletionWriter,
) -> Unit raise {
  writer.append_prefill_completed(
    slot=slot(0),
    request_id=request_id(7UL),
    request_generation=@inference.RequestGeneration::initial(),
    processed_tokens=2,
  )
  writer.append_decode_completed(
    slot=slot(1),
    request_id=request_id(8UL),
    request_generation=@inference.RequestGeneration::from_wire(3UL),
    token_id=42,
  )
}

///|
test "worker lease writes and submits without exposing mutable owners" {
  let plan_owner = @worker_protocol.SchedulePlanBuffer::new(protocol_limits())
  let plan = submitted_plan_for_completion(plan_owner)
  let completions = @worker_protocol.CompletionBuffer::new(protocol_limits())
  let writer = completions.writer_for(plan)
  append_mixed_writer(writer)
  let submitted = writer.submit()
  plan_owner.validate_submitted(plan)
  completions.validate_submitted(submitted, plan)
  assert_eq(submitted.sequence_wire(), plan.sequence_wire())
  assert_eq(
    submitted.model_plan_generation_wire(),
    plan.model_plan_generation_wire(),
  )
  assert_eq(submitted.entry_count(), 2)
  assert_eq(submitted.entry(1).token_id(), 42)
  let consumed = completions.consume(submitted, plan)
  let retired = plan_owner.retire(plan)
  completions.reset(consumed)
  plan_owner.reset(retired)
  try writer.abort() catch {
    StaleCompletionBufferEpoch => ()
    _ => fail("expected reset writer to stay stale")
  } noraise {
    _ => fail("expected reset writer abort to fail")
  }
}

///|
test "partial completion abort stales writer and permits exact retry" {
  let plan_owner = @worker_protocol.SchedulePlanBuffer::new(protocol_limits())
  let plan = submitted_plan_for_completion(plan_owner)
  let completions = @worker_protocol.CompletionBuffer::new(protocol_limits())
  let partial = completions.writer_for(plan)
  try completions.writer_for(plan) catch {
    CompletionWriterAlreadyOpen => ()
    _ => fail("expected exclusive writer rejection")
  } noraise {
    _ => fail("expected duplicate writer issuance to fail")
  }
  try
    completions.append_failed(
      slot=slot(1),
      request_id=request_id(8UL),
      request_generation=@inference.RequestGeneration::from_wire(3UL),
      failure=CancelledBeforeExecution,
    )
  catch {
    CompletionWriterAlreadyOpen => ()
    _ => fail("expected direct owner mutation rejection")
  } noraise {
    _ => fail("expected direct owner append to fail")
  }
  try completions.submit_for(plan) catch {
    CompletionWriterAlreadyOpen => ()
    _ => fail("expected direct owner submit rejection")
  } noraise {
    _ => fail("expected direct owner submit to fail")
  }
  partial.append_prefill_completed(
    slot=slot(0),
    request_id=request_id(7UL),
    request_generation=@inference.RequestGeneration::initial(),
    processed_tokens=2,
  )
  try partial.submit() catch {
    CompletionCountMismatch => ()
    _ => fail("expected partial completion rejection")
  } noraise {
    _ => fail("expected submit to fail")
  }
  partial.abort()
  try
    partial.append_decode_completed(
      slot=slot(1),
      request_id=request_id(8UL),
      request_generation=@inference.RequestGeneration::from_wire(3UL),
      token_id=42,
    )
  catch {
    StaleCompletionBufferEpoch => ()
    _ => fail("expected aborted writer to stay stale")
  } noraise {
    _ => fail("expected stale append to fail")
  }
  let retry = completions.writer_for(plan)
  append_mixed_writer(retry)
  let submitted = retry.submit()
  completions.validate_submitted(submitted, plan)
  assert_eq(submitted.entry_count(), 2)
}

///|
test "nonmutating owner preflight rejects foreign handles" {
  let left_plan_owner = @worker_protocol.SchedulePlanBuffer::new(
    protocol_limits(),
  )
  let right_plan_owner = @worker_protocol.SchedulePlanBuffer::new(
    protocol_limits(),
  )
  let left_plan = submitted_plan_for_completion(left_plan_owner)
  let right_plan = submitted_plan_for_completion(right_plan_owner)
  try left_plan_owner.validate_submitted(right_plan) catch {
    PlanBufferMismatch => ()
    _ => fail("expected foreign plan owner rejection")
  } noraise {
    _ => fail("expected plan preflight to fail")
  }
  let left_completions = @worker_protocol.CompletionBuffer::new(
    protocol_limits(),
  )
  let right_completions = @worker_protocol.CompletionBuffer::new(
    protocol_limits(),
  )
  let left_writer = left_completions.writer_for(left_plan)
  append_mixed_writer(left_writer)
  let left_submitted = left_writer.submit()
  try right_completions.validate_submitted(left_submitted, left_plan) catch {
    CompletionBufferMismatch => ()
    _ => fail("expected foreign completion owner rejection")
  } noraise {
    _ => fail("expected completion preflight to fail")
  }
}

///|
test "worker lease supports final-prefill and failed scalar outcomes" {
  let final_plan_owner = @worker_protocol.SchedulePlanBuffer::new(
    protocol_limits(),
  )
  final_plan_owner.append_final_prefill(
    request_id=request_id(17UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_start=0,
    tokens=[31, 32],
    pages=[allocator().allocate_one()],
    capabilities=[@plan.KernelCapabilityId::embedding_lookup()],
    sampling=@inference.SamplingParameters::greedy(),
    completion_slot=slot(0),
  )
  let final_plan = final_plan_owner.submit_wire(
    sequence_value=1UL,
    model_plan_generation_value=1UL,
    token_budget=2,
    has_previous=false,
    previous_sequence_value=0UL,
    loaded_generation_value=1UL,
  )
  let final_completions = @worker_protocol.CompletionBuffer::new(
    protocol_limits(),
  )
  let final_writer = final_completions.writer_for(final_plan)
  final_writer.append_final_prefill_completed(
    slot=slot(0),
    request_id=request_id(17UL),
    request_generation=@inference.RequestGeneration::initial(),
    processed_tokens=2,
    token_id=91,
  )
  let final_submitted = final_writer.submit()
  assert_eq(final_submitted.entry(0).token_id(), 91)

  let failed_plan_owner = @worker_protocol.SchedulePlanBuffer::new(
    protocol_limits(),
  )
  let failed_plan = submitted_plan_for_completion(failed_plan_owner)
  let failed_completions = @worker_protocol.CompletionBuffer::new(
    protocol_limits(),
  )
  let failed_writer = failed_completions.writer_for(failed_plan)
  failed_writer.append_failed(
    slot=slot(0),
    request_id=request_id(7UL),
    request_generation=@inference.RequestGeneration::initial(),
    failure=ExecutionFailed,
  )
  failed_writer.append_failed(
    slot=slot(1),
    request_id=request_id(8UL),
    request_generation=@inference.RequestGeneration::from_wire(3UL),
    failure=ResourceUnavailable,
  )
  let failed_submitted = failed_writer.submit()
  assert_eq(failed_submitted.entry(0).failure(), ExecutionFailed)
  assert_eq(failed_submitted.entry(1).failure(), ResourceUnavailable)
}

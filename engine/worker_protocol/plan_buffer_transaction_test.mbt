///|
test "failed appends leave every logical count unchanged" {
  let limits = protocol_limits()
  let buffer = @worker_protocol.SchedulePlanBuffer::new(limits)
  let pages = allocator()
  buffer.append_prefill(
    request_id=request_id(1UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_start=0,
    tokens=[10, 11],
    pages=[pages.allocate_one()],
    capabilities=[@plan.KernelCapabilityId::embedding_lookup()],
    completion_slot=slot(0),
  )
  try
    buffer.append_decode(
      request_id=request_id(2UL),
      request_generation=@inference.RequestGeneration::initial(),
      sequence_token_position=2,
      input_token=1025,
      pages=[pages.allocate_one()],
      capabilities=[@plan.KernelCapabilityId::causal_attention()],
      sampling=@inference.SamplingParameters::greedy(),
      completion_slot=slot(1),
    )
  catch {
    @worker_protocol.InvalidToken(index=2) => ()
    _ => fail("expected invalid token")
  } noraise {
    _ => fail("expected append to fail")
  }
  buffer.append_decode(
    request_id=request_id(2UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_position=2,
    input_token=12,
    pages=[pages.allocate_one()],
    capabilities=[@plan.KernelCapabilityId::causal_attention()],
    sampling=@inference.SamplingParameters::greedy(),
    completion_slot=slot(1),
  )
  let submitted = submit_mixed_buffer(
    buffer,
    @worker_protocol.PlanSequence::first(),
    None,
  )
  debug_inspect(
    (
      submitted.token_count(),
      submitted.page_count(),
      submitted.capability_count(),
      submitted.row_count(),
    ),
    content="(3, 2, 2, 2)",
  )
}

///|
test "overflow duplicate and row ordering failures roll back exactly" {
  let limits = try! @worker_protocol.WorkerProtocolLimits::new(
    max_prefill_rows=1,
    max_decode_rows=1,
    max_plan_rows=2,
    max_plan_tokens=2,
    max_plan_pages=2,
    max_capabilities=2,
    max_completion_slots=2,
    max_sequence_tokens=8,
    max_token_id=64,
  )
  let buffer = @worker_protocol.SchedulePlanBuffer::new(limits)
  let pages = allocator()
  buffer.append_decode(
    request_id=request_id(1UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_position=0,
    input_token=1,
    pages=[pages.allocate_one()],
    capabilities=[@plan.KernelCapabilityId::causal_attention()],
    sampling=@inference.SamplingParameters::greedy(),
    completion_slot=try! @worker_protocol.CompletionSlot::new(0, limits),
  )
  try
    buffer.append_prefill(
      request_id=request_id(2UL),
      request_generation=@inference.RequestGeneration::initial(),
      sequence_token_start=0,
      tokens=[2],
      pages=[pages.allocate_one()],
      capabilities=[@plan.KernelCapabilityId::embedding_lookup()],
      completion_slot=try! @worker_protocol.CompletionSlot::new(1, limits),
    )
  catch {
    @worker_protocol.InvalidRowDescriptor(kind=Prefill, rule=InvalidRowOrder) =>
      ()
    _ => fail("expected prefill after decode rejection")
  } noraise {
    _ => fail("expected append to fail")
  }
  try
    buffer.append_decode(
      request_id=request_id(1UL),
      request_generation=@inference.RequestGeneration::initial(),
      sequence_token_position=1,
      input_token=2,
      pages=[pages.allocate_one()],
      capabilities=[@plan.KernelCapabilityId::causal_attention()],
      sampling=@inference.SamplingParameters::greedy(),
      completion_slot=try! @worker_protocol.CompletionSlot::new(1, limits),
    )
  catch {
    @worker_protocol.BoundExceeded(DecodeRows) => ()
    _ => fail("expected decode row capacity")
  } noraise {
    _ => fail("expected append to fail")
  }
  let submitted = buffer.submit(
    sequence=@worker_protocol.PlanSequence::first(),
    model_plan_generation=model_generation(1UL),
    token_budget=1,
    None,
    model_generation(1UL),
  )
  debug_inspect(
    (submitted.prefill_count(), submitted.decode_count(), submitted.token_at(0)),
    content="(0, 1, 1)",
  )
}

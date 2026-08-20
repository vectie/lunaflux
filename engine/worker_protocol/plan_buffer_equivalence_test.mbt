///|
test "submitted buffer exposes exact canonical plan fields" {
  let buffer = @worker_protocol.SchedulePlanBuffer::new(protocol_limits())
  append_mixed_buffer(buffer, allocator())
  let actual = submit_mixed_buffer(
    buffer,
    plan_sequence(2UL),
    Some(plan_sequence(1UL)),
  )
  assert_eq(actual.sequence(), plan_sequence(2UL))
  assert_eq(actual.model_plan_generation(), model_generation(1UL))
  assert_eq(actual.token_budget(), 3)
  assert_eq(actual.token_count(), 3)
  assert_eq(actual.token_at(0), 10)
  assert_eq(actual.token_at(1), 11)
  assert_eq(actual.token_at(2), 12)
  assert_eq(actual.page_count(), 2)
  assert_eq(actual.page_at(0).index(), 0)
  assert_eq(actual.page_at(1).index(), 1)
  assert_eq(actual.capability_count(), 2)
  assert_eq(
    actual.capability_at(0),
    @plan.KernelCapabilityId::embedding_lookup(),
  )
  assert_eq(
    actual.capability_at(1),
    @plan.KernelCapabilityId::causal_attention(),
  )
  let actual_prefill = actual.prefill_row(0)
  assert_eq(actual_prefill.request_id(), request_id(7UL))
  assert_eq(
    actual_prefill.request_generation(),
    @inference.RequestGeneration::initial(),
  )
  assert_eq(actual_prefill.sequence_token_start(), 0)
  assert_eq(actual_prefill.tokens(), token_range(0, 2))
  assert_eq(actual_prefill.page_table(), page_view(0, 1))
  assert_eq(actual_prefill.capabilities(), capability_view(0, 1))
  assert_false(actual_prefill.produces_token())
  assert_eq(actual_prefill.completion_slot(), slot(0))
  let actual_decode = actual.decode_row(0)
  assert_eq(actual_decode.request_id(), request_id(8UL))
  assert_eq(
    actual_decode.request_generation(),
    @inference.RequestGeneration::from_wire(3UL),
  )
  assert_eq(actual_decode.sequence_token_position(), 3)
  assert_eq(actual_decode.input_token(), token_range(2, 1))
  assert_eq(actual_decode.page_table(), page_view(1, 1))
  assert_eq(actual_decode.capabilities(), capability_view(1, 1))
  assert_eq(actual_decode.sampling(), @inference.SamplingParameters::greedy())
  assert_eq(actual_decode.completion_slot(), slot(1))
}

///|
test "append copies caller table views before return" {
  let buffer = @worker_protocol.SchedulePlanBuffer::new(protocol_limits())
  let pages = allocator()
  let input_tokens = [10, 11]
  let input_pages = [pages.allocate_one()]
  let original_page = input_pages[0]
  let input_capabilities = [@plan.KernelCapabilityId::embedding_lookup()]
  buffer.append_prefill(
    request_id=request_id(1UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_start=0,
    tokens=input_tokens,
    pages=input_pages,
    capabilities=input_capabilities,
    completion_slot=slot(0),
  )
  input_tokens[0] = 99
  input_pages[0] = pages.allocate_one()
  input_capabilities[0] = @plan.KernelCapabilityId::causal_attention()
  let submitted = buffer.submit(
    sequence=@worker_protocol.PlanSequence::first(),
    model_plan_generation=model_generation(1UL),
    token_budget=2,
    None,
    model_generation(1UL),
  )
  assert_eq(submitted.token_at(0), 10)
  assert_eq(submitted.page_at(0), original_page)
  assert_eq(
    submitted.capability_at(0),
    @plan.KernelCapabilityId::embedding_lookup(),
  )
}

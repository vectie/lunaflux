///|
test "submitted buffer is exactly equivalent to immutable plan fixture" {
  let expected = mixed_plan()
  let buffer = @worker_protocol.SchedulePlanBuffer::new(protocol_limits())
  append_mixed_buffer(buffer, allocator())
  let actual = submit_mixed_buffer(
    buffer,
    plan_sequence(2UL),
    Some(plan_sequence(1UL)),
  )
  assert_eq(actual.sequence(), expected.sequence())
  assert_eq(actual.model_plan_generation(), expected.model_plan_generation())
  assert_eq(actual.token_budget(), expected.token_budget())
  assert_eq(actual.token_count(), expected.tokens().length())
  for index, token in expected.tokens() {
    assert_eq(actual.token_at(index), token)
  }
  assert_eq(actual.page_count(), expected.pages().length())
  for index, page in expected.pages() {
    assert_eq(actual.page_at(index), page)
  }
  assert_eq(actual.capability_count(), expected.capabilities().length())
  for index, capability in expected.capabilities() {
    assert_eq(actual.capability_at(index), capability)
  }
  let expected_prefill = expected.prefill_rows()[0]
  let actual_prefill = actual.prefill_row(0)
  assert_eq(actual_prefill.request_id(), expected_prefill.request_id())
  assert_eq(
    actual_prefill.request_generation(),
    expected_prefill.request_generation(),
  )
  assert_eq(
    actual_prefill.sequence_token_start(),
    expected_prefill.sequence_token_start(),
  )
  assert_eq(actual_prefill.tokens(), expected_prefill.tokens())
  assert_eq(actual_prefill.page_table(), expected_prefill.page_table())
  assert_eq(actual_prefill.capabilities(), expected_prefill.capabilities())
  assert_eq(
    actual_prefill.completion_slot(),
    expected_prefill.completion_slot(),
  )
  let expected_decode = expected.decode_rows()[0]
  let actual_decode = actual.decode_row(0)
  assert_eq(actual_decode.request_id(), expected_decode.request_id())
  assert_eq(
    actual_decode.request_generation(),
    expected_decode.request_generation(),
  )
  assert_eq(
    actual_decode.sequence_token_position(),
    expected_decode.sequence_token_position(),
  )
  assert_eq(actual_decode.input_token(), expected_decode.input_token())
  assert_eq(actual_decode.page_table(), expected_decode.page_table())
  assert_eq(actual_decode.capabilities(), expected_decode.capabilities())
  assert_eq(actual_decode.sampling(), expected_decode.sampling())
  assert_eq(actual_decode.completion_slot(), expected_decode.completion_slot())
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

///|
test "scalar row drafts equal exact immutable plan semantics" {
  let pages = allocator()
  let page0 = pages.allocate_one()
  let page1 = pages.allocate_one()
  let buffer = @worker_protocol.SchedulePlanBuffer::new(protocol_limits())
  let prefill = buffer.begin_prefill(
    request_id=request_id(7UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_start=0,
    completion_slot=slot(0),
  )
  prefill.push_token(10)
  prefill.push_token(11)
  prefill.push_page(page0)
  prefill.push_capability(@plan.KernelCapabilityId::embedding_lookup())
  prefill.commit()
  let decode = buffer.begin_decode(
    request_id=request_id(8UL),
    request_generation=@inference.RequestGeneration::from_wire(3UL),
    sequence_token_position=3,
    sampling=@inference.SamplingParameters::greedy(),
    completion_slot=slot(1),
  )
  decode.push_token(12)
  decode.push_page(page1)
  decode.push_capability(@plan.KernelCapabilityId::causal_attention())
  decode.commit()
  let actual = submit_mixed_buffer(
    buffer,
    plan_sequence(2UL),
    Some(plan_sequence(1UL)),
  )
  assert_eq(actual.token_at(0), 10)
  assert_eq(actual.token_at(1), 11)
  assert_eq(actual.token_at(2), 12)
  assert_eq(actual.page_at(0), page0)
  assert_eq(actual.page_at(1), page1)
  assert_eq(actual.prefill_row(0).tokens(), token_range(0, 2))
  assert_eq(actual.decode_row(0).input_token(), token_range(2, 1))
  assert_eq(actual.prefill_row(0).page_table(), page_view(0, 1))
  assert_eq(actual.decode_row(0).page_table(), page_view(1, 1))
}

///|
test "draft rollback restores exact suffix after every construction stage" {
  let buffer = @worker_protocol.SchedulePlanBuffer::new(protocol_limits())
  let pages = allocator()
  for stage in 0..<4 {
    let draft = buffer.begin_prefill(
      request_id=request_id((stage + 1).to_uint64()),
      request_generation=@inference.RequestGeneration::initial(),
      sequence_token_start=0,
      completion_slot=slot(stage),
    )
    if stage >= 1 {
      draft.push_token(10 + stage)
    }
    if stage >= 2 {
      draft.push_page(pages.allocate_one())
    }
    if stage >= 3 {
      draft.push_capability(@plan.KernelCapabilityId::embedding_lookup())
    }
    draft.rollback()
  }
  let last_draft = buffer.begin_decode(
    request_id=request_id(9UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_position=0,
    sampling=@inference.SamplingParameters::greedy(),
    completion_slot=slot(0),
  )
  last_draft.push_token(7)
  last_draft.push_page(pages.allocate_one())
  last_draft.push_capability(@plan.KernelCapabilityId::causal_attention())
  last_draft.commit()
  let submitted = buffer.submit(
    sequence=@worker_protocol.PlanSequence::first(),
    model_plan_generation=model_generation(1UL),
    token_budget=1,
    None,
    model_generation(1UL),
  )
  assert_eq(submitted.row_count(), 1)
  assert_eq(submitted.token_count(), 1)
  assert_eq(submitted.page_count(), 1)
  assert_eq(submitted.capability_count(), 1)
  assert_eq(submitted.token_at(0), 7)
}

///|
test "draft handles reject overlapping stale and closed mutations" {
  let buffer = @worker_protocol.SchedulePlanBuffer::new(protocol_limits())
  let other = @worker_protocol.SchedulePlanBuffer::new(protocol_limits())
  let old = buffer.begin_prefill(
    request_id=request_id(1UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_start=0,
    completion_slot=slot(0),
  )
  try
    buffer.begin_decode(
      request_id=request_id(2UL),
      request_generation=@inference.RequestGeneration::initial(),
      sequence_token_position=0,
      sampling=@inference.SamplingParameters::greedy(),
      completion_slot=slot(1),
    )
  catch {
    PlanRowDraftAlreadyOpen => ()
    _ => fail("expected one open draft")
  } noraise {
    _ => fail("expected begin to fail")
  }
  try other.rollback_draft(old) catch {
    PlanRowDraftMismatch => ()
    _ => fail("expected wrong draft owner")
  } noraise {
    _ => fail("expected rollback to fail")
  }
  buffer.rollback_draft(old)
  let current = buffer.begin_decode(
    request_id=request_id(2UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_position=0,
    sampling=@inference.SamplingParameters::greedy(),
    completion_slot=slot(1),
  )
  try old.push_token(3) catch {
    PlanRowDraftMismatch => ()
    _ => fail("expected stale draft")
  } noraise {
    _ => fail("expected push to fail")
  }
  current.rollback()
}

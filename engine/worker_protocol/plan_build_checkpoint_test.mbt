///|
fn append_checkpoint_prefill(
  buffer : @worker_protocol.SchedulePlanBuffer,
  pages : @page_allocator.PageAllocator,
  id : UInt64,
  token : Int,
  completion_slot : Int,
) -> Unit raise {
  buffer.append_prefill(
    request_id=request_id(id),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_start=0,
    tokens=[token],
    pages=[pages.allocate_one()],
    capabilities=[@plan.KernelCapabilityId::embedding_lookup()],
    completion_slot=slot(completion_slot),
  )
}

///|
test "build checkpoint rolls back several committed rows for exact retry" {
  let buffer = @worker_protocol.SchedulePlanBuffer::new(protocol_limits())
  let pages = allocator()
  let checkpoint = buffer.checkpoint()
  append_checkpoint_prefill(buffer, pages, 1UL, 10, 0)
  append_checkpoint_prefill(buffer, pages, 2UL, 11, 1)
  let draft = buffer.begin_decode(
    request_id=request_id(3UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_position=1,
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=sampling_seed(),
    sample_index=0,
    completion_slot=slot(2),
  )
  draft.push_token(12)
  draft.push_page(pages.allocate_one())
  draft.push_capability(@plan.KernelCapabilityId::causal_attention())
  draft.commit()
  buffer.rollback_to(checkpoint)

  append_checkpoint_prefill(buffer, pages, 9UL, 20, 0)
  let retry = buffer.begin_decode(
    request_id=request_id(10UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_position=1,
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=sampling_seed(),
    sample_index=0,
    completion_slot=slot(1),
  )
  retry.push_token(21)
  retry.push_page(pages.allocate_one())
  retry.push_capability(@plan.KernelCapabilityId::causal_attention())
  retry.commit()
  let submitted = buffer.submit(
    sequence=@worker_protocol.PlanSequence::first(),
    model_plan_generation=model_generation(1UL),
    token_budget=2,
    None,
    model_generation(1UL),
  )
  assert_eq(submitted.row_count(), 2)
  assert_eq(submitted.token_count(), 2)
  assert_eq(submitted.page_count(), 2)
  assert_eq(submitted.capability_count(), 2)
  assert_eq(submitted.token_at(0), 20)
  assert_eq(submitted.token_at(1), 21)
  assert_eq(submitted.prefill_row(0).request_id(), request_id(9UL))
  assert_eq(submitted.decode_row(0).request_id(), request_id(10UL))
}

///|
test "checkpoint rejects wrong owner superseded handle and open draft" {
  let left = @worker_protocol.SchedulePlanBuffer::new(protocol_limits())
  let right = @worker_protocol.SchedulePlanBuffer::new(protocol_limits())
  let old = left.checkpoint()
  try right.rollback_to(old) catch {
    PlanBuildCheckpointMismatch => ()
    _ => fail("expected checkpoint owner mismatch")
  } noraise {
    _ => fail("expected rollback to fail")
  }
  let current = left.checkpoint()
  try left.rollback_to(old) catch {
    StalePlanBuildCheckpointEpoch => ()
    _ => fail("expected superseded checkpoint")
  } noraise {
    _ => fail("expected rollback to fail")
  }
  let draft = left.begin_prefill(
    request_id=request_id(1UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_start=0,
    completion_slot=slot(0),
  )
  try left.rollback_to(current) catch {
    PlanRowDraftAlreadyOpen => ()
    _ => fail("expected explicit open draft rejection")
  } noraise {
    _ => fail("expected rollback to fail")
  }
  draft.rollback()
  left.rollback_to(current)
  try left.rollback_to(current) catch {
    StalePlanBuildCheckpointEpoch => ()
    _ => fail("expected consumed checkpoint")
  } noraise {
    _ => fail("expected rollback to fail")
  }
}

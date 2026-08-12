///|
test "rollback clears discarded page cells" {
  let limits = WorkerProtocolLimits::new(
    max_prefill_rows=2,
    max_decode_rows=1,
    max_plan_rows=3,
    max_plan_tokens=3,
    max_plan_pages=3,
    max_capabilities=3,
    max_completion_slots=3,
    max_sequence_tokens=8,
    max_token_id=64,
  )
  let allocator_limits = @page_allocator.PageAllocatorLimits::new(
    page_capacity=3,
    max_run_pages=3,
    max_active_references_per_page=1,
    max_cached_references_per_page=1,
  )
  let allocator = @page_allocator.PageAllocator::new(allocator_limits)
  let request_id = @inference.RequestId::new(1UL)
  let slot = CompletionSlot::new(0, limits)
  let buffer = SchedulePlanBuffer::new(limits)
  let checkpoint = buffer.checkpoint()
  buffer.append_prefill(
    request_id~,
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_start=0,
    tokens=[1],
    pages=[allocator.allocate_one()],
    capabilities=[@plan.KernelCapabilityId::embedding_lookup()],
    completion_slot=slot,
  )
  assert_true(buffer.pages.is_occupied(0))
  buffer.rollback_to(checkpoint)
  assert_false(buffer.pages.is_occupied(0))
  assert_eq(buffer.token_count, 0)
  assert_eq(buffer.page_count, 0)
  assert_eq(buffer.capability_count, 0)
  assert_eq(buffer.prefill_count, 0)
  assert_eq(buffer.decode_count, 0)
}

///|
test "checkpoint overflow fails before mutating snapshot identity" {
  let limits = WorkerProtocolLimits::new(
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
  let buffer = SchedulePlanBuffer::new(limits)
  buffer.checkpoint_epoch = 0xffffffffffffffffUL
  try buffer.checkpoint() catch {
    IdentityOverflow(PlanBuildCheckpointEpochIdentity) => ()
    _ => fail("expected checkpoint epoch overflow")
  } noraise {
    _ => fail("expected checkpoint to fail")
  }
  assert_eq(buffer.checkpoint_epoch, 0xffffffffffffffffUL)
  assert_eq(buffer.token_count, 0)
  assert_eq(buffer.page_count, 0)
}

///|
fn fixture_identity() -> @spec.ModelIdentity {
  let content = try! @spec.ContentDigest::from_sha256(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  let plan = try! @spec.PlanDigest::from_sha256(
    "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
  )
  try! @spec.ModelIdentity::new(content, plan)
}

///|
fn fixture_limits() -> @worker_protocol.WorkerProtocolLimits {
  try! @worker_protocol.WorkerProtocolLimits::new(
    max_prefill_rows=1,
    max_decode_rows=1,
    max_plan_rows=1,
    max_plan_tokens=2,
    max_plan_pages=1,
    max_capabilities=1,
    max_completion_slots=1,
    max_sequence_tokens=8,
    max_token_id=128,
  )
}

///|
fn fixture_allocator() -> @page_allocator.PageAllocator {
  let limits = try! @page_allocator.PageAllocatorLimits::new(
    page_capacity=2,
    max_run_pages=2,
    max_active_references_per_page=2,
    max_cached_references_per_page=2,
  )
  @page_allocator.PageAllocator::new(limits)
}

///|
fn fixture_request_id() -> @inference.RequestId {
  try! @inference.RequestId::new(7UL)
}

///|
fn fixture_plan(
  sequence : UInt64,
  previous : UInt64,
) -> (
  @worker_protocol.SchedulePlanBuffer,
  @worker_protocol.SubmittedSchedulePlan,
) {
  let owner = try! @worker_protocol.SchedulePlanBuffer::new(fixture_limits())
  try! owner.append_prefill(
    request_id=fixture_request_id(),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_start=0,
    tokens=[3],
    pages=[try! fixture_allocator().allocate_one()],
    capabilities=[@plan.KernelCapabilityId::embedding_lookup()],
    completion_slot=try! @worker_protocol.CompletionSlot::new(
      0,
      fixture_limits(),
    ),
  )
  let submitted = try! owner.submit_wire(
    sequence_value=sequence,
    model_plan_generation_value=1UL,
    token_budget=1,
    has_previous=previous != 0UL,
    previous_sequence_value=previous,
    loaded_generation_value=1UL,
  )
  (owner, submitted)
}

///|
fn fixture_completion(
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> (@worker_protocol.CompletionBuffer, @worker_protocol.SubmittedCompletion) {
  let owner = try! @worker_protocol.CompletionBuffer::new(fixture_limits())
  try! owner.append_prefill_completed(
    slot=try! @worker_protocol.CompletionSlot::new(0, fixture_limits()),
    request_id=fixture_request_id(),
    request_generation=@inference.RequestGeneration::initial(),
    processed_tokens=1,
  )
  (owner, try! owner.submit_for(plan))
}

///|
fn fixture_group(
  world_size : Int,
  plan_buffer : @worker_protocol.SchedulePlanBuffer,
) -> @rank_group_protocol.RankGroupOwner {
  try! @rank_group_protocol.RankGroupOwner::new(
    identity=fixture_identity(),
    model_plan_generation=@worker_protocol.ModelPlanGeneration::first(),
    generation=@rank_group_protocol.RankGroupGeneration::first(),
    world_size~,
    previous_plan_sequence_value=0UL,
    plan_buffer_identities=[plan_buffer.identity()],
  )
}

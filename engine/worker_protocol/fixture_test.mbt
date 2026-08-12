///|
fn protocol_limits() -> @worker_protocol.WorkerProtocolLimits {
  try! @worker_protocol.WorkerProtocolLimits::new(
    max_prefill_rows=4,
    max_decode_rows=4,
    max_plan_rows=8,
    max_plan_tokens=32,
    max_plan_pages=32,
    max_capabilities=32,
    max_completion_slots=8,
    max_sequence_tokens=64,
    max_token_id=1024,
  )
}

///|
fn allocator() -> @page_allocator.PageAllocator {
  let limits = try! @page_allocator.PageAllocatorLimits::new(
    page_capacity=8,
    max_run_pages=8,
    max_active_references_per_page=8,
    max_cached_references_per_page=8,
  )
  @page_allocator.PageAllocator::new(limits)
}

///|
fn request_id(value : UInt64) -> @inference.RequestId {
  try! @inference.RequestId::new(value)
}

///|
fn plan_sequence(value : UInt64) -> @worker_protocol.PlanSequence {
  try! @worker_protocol.PlanSequence::from_wire(value)
}

///|
fn model_generation(value : UInt64) -> @worker_protocol.ModelPlanGeneration {
  try! @worker_protocol.ModelPlanGeneration::from_wire(value)
}

///|
fn token_range(start : Int, length : Int) -> @worker_protocol.TokenRange {
  try! @worker_protocol.TokenRange::new(start~, length~, protocol_limits())
}

///|
fn page_view(start : Int, length : Int) -> @worker_protocol.PageTableView {
  try! @worker_protocol.PageTableView::new(start~, length~, protocol_limits())
}

///|
fn capability_view(
  start : Int,
  length : Int,
) -> @worker_protocol.CapabilityView {
  try! @worker_protocol.CapabilityView::new(start~, length~, protocol_limits())
}

///|
fn slot(value : Int) -> @worker_protocol.CompletionSlot {
  try! @worker_protocol.CompletionSlot::new(value, protocol_limits())
}

///|
fn prefill(
  id : UInt64,
  generation : UInt64,
  token_start : Int,
  token_length : Int,
  page_start : Int,
  capability_start : Int,
  completion_slot : Int,
) -> @worker_protocol.PrefillRow {
  try! @worker_protocol.PrefillRow::new(
    request_id=request_id(id),
    request_generation=@inference.RequestGeneration::from_wire(generation),
    sequence_token_start=0,
    tokens=token_range(token_start, token_length),
    page_table=page_view(page_start, 1),
    capabilities=capability_view(capability_start, 1),
    completion_slot=slot(completion_slot),
    limits=protocol_limits(),
  )
}

///|
fn decode(
  id : UInt64,
  generation : UInt64,
  token_start : Int,
  page_start : Int,
  capability_start : Int,
  completion_slot : Int,
) -> @worker_protocol.DecodeRow {
  try! @worker_protocol.DecodeRow::new(
    request_id=request_id(id),
    request_generation=@inference.RequestGeneration::from_wire(generation),
    sequence_token_position=3,
    input_token=token_range(token_start, 1),
    page_table=page_view(page_start, 1),
    capabilities=capability_view(capability_start, 1),
    sampling=@inference.SamplingParameters::greedy(),
    completion_slot=slot(completion_slot),
    limits=protocol_limits(),
  )
}

///|
fn mixed_plan() -> @worker_protocol.SchedulePlan {
  let pages = allocator()
  try! @worker_protocol.SchedulePlan::new(
    sequence=plan_sequence(2UL),
    model_plan_generation=model_generation(1UL),
    token_budget=3,
    tokens=[10, 11, 12],
    pages=[pages.allocate_one(), pages.allocate_one()],
    capabilities=[
      @plan.KernelCapabilityId::embedding_lookup(),
      @plan.KernelCapabilityId::causal_attention(),
    ],
    prefill_rows=[prefill(7UL, 0UL, 0, 2, 0, 0, 0)],
    decode_rows=[decode(8UL, 3UL, 2, 1, 1, 1)],
    limits=protocol_limits(),
  )
}

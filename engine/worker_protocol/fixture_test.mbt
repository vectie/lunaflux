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
fn sampling_seed() -> @inference.SamplingSeed {
  try! @inference.SamplingSeed::new(0x123456789abcdef0UL)
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

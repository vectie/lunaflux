///|
fn wire_protocol_limits() -> @worker_protocol.WorkerProtocolLimits {
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
fn wire_inference_limits() -> @inference.InferenceLimits {
  try! @inference.InferenceLimits::new(
    max_text_bytes=1024,
    max_input_tokens=64,
    max_new_tokens=64,
    max_context_tokens=64,
    max_token_id=1024,
    max_stop_token_ids=8,
    max_stop_strings=1,
    max_stop_string_bytes=1,
    max_trace_bytes=64,
    max_cache_scope_bytes=64,
    max_decoded_delta_bytes=64,
    max_deadline_millis=1000UL,
    max_top_k=32,
    max_temperature=4.0,
  )
}

///|
fn wire_allocator() -> @page_allocator.PageAllocator {
  @page_allocator.PageAllocator::new(
    try! @page_allocator.PageAllocatorLimits::new(
      page_capacity=8,
      max_run_pages=8,
      max_active_references_per_page=8,
      max_cached_references_per_page=8,
    ),
  )
}

///|
fn wire_request(value : UInt64) -> @inference.RequestId {
  try! @inference.RequestId::new(value)
}

///|
fn wire_slot(value : Int) -> @worker_protocol.CompletionSlot {
  try! @worker_protocol.CompletionSlot::new(value, wire_protocol_limits())
}

///|
fn wire_sampling() -> @inference.SamplingParameters {
  try! @inference.SamplingParameters::sample(
    temperature=0.75,
    top_k=16,
    top_p=0.9,
    wire_inference_limits(),
  )
}

///|
fn wire_seed(value : UInt64) -> @inference.SamplingSeed {
  try! @inference.SamplingSeed::new(value)
}

///|
fn submitted_wire_fixture() -> @worker_protocol.SubmittedSchedulePlan raise {
  let limits = wire_protocol_limits()
  let owner = @worker_protocol.SchedulePlanBuffer::new(limits)
  let pages = wire_allocator()
  owner.append_final_prefill(
    request_id=wire_request(7UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_start=3,
    tokens=[10, 11],
    pages=[pages.allocate_one()],
    capabilities=[@plan.KernelCapabilityId::positioned_rotary_embedding()],
    sampling=wire_sampling(),
    sampling_seed=wire_seed(0x1234UL),
    sample_index=5,
    completion_slot=wire_slot(0),
  )
  owner.append_decode(
    request_id=wire_request(8UL),
    request_generation=@inference.RequestGeneration::from_wire(3UL),
    sequence_token_position=6,
    input_token=12,
    pages=[pages.allocate_one()],
    capabilities=[@plan.KernelCapabilityId::paged_causal_attention()],
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=wire_seed(0x5678UL),
    sample_index=6,
    completion_slot=wire_slot(1),
  )
  owner.submit_wire(
    sequence_value=1UL,
    model_plan_generation_value=9UL,
    token_budget=3,
    has_previous=false,
    previous_sequence_value=0UL,
    loaded_generation_value=9UL,
  )
}

///|
fn submitted_completion_fixture(
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> @worker_protocol.SubmittedCompletion raise {
  let owner = @worker_protocol.CompletionBuffer::new(wire_protocol_limits())
  owner.append_final_prefill_completed(
    slot=wire_slot(0),
    request_id=wire_request(7UL),
    request_generation=@inference.RequestGeneration::initial(),
    processed_tokens=2,
    token_id=17,
  )
  owner.append_decode_completed(
    slot=wire_slot(1),
    request_id=wire_request(8UL),
    request_generation=@inference.RequestGeneration::from_wire(3UL),
    token_id=18,
  )
  owner.submit_for(plan)
}

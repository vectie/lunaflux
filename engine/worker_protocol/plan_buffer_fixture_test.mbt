///|
fn append_mixed_buffer(
  buffer : @worker_protocol.SchedulePlanBuffer,
  pages : @page_allocator.PageAllocator,
) -> Unit raise {
  buffer.append_prefill(
    request_id=request_id(7UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_start=0,
    tokens=[10, 11],
    pages=[pages.allocate_one()],
    capabilities=[@plan.KernelCapabilityId::embedding_lookup()],
    completion_slot=slot(0),
  )
  buffer.append_decode(
    request_id=request_id(8UL),
    request_generation=@inference.RequestGeneration::from_wire(3UL),
    sequence_token_position=3,
    input_token=12,
    pages=[pages.allocate_one()],
    capabilities=[@plan.KernelCapabilityId::causal_attention()],
    sampling=@inference.SamplingParameters::greedy(),
    completion_slot=slot(1),
  )
}

///|
fn submit_mixed_buffer(
  buffer : @worker_protocol.SchedulePlanBuffer,
  sequence : @worker_protocol.PlanSequence,
  previous_sequence : @worker_protocol.PlanSequence?,
) -> @worker_protocol.SubmittedSchedulePlan raise {
  buffer.submit(
    sequence~,
    model_plan_generation=model_generation(1UL),
    token_budget=3,
    previous_sequence,
    model_generation(1UL),
  )
}

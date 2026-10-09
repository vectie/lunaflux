///|
/// Offline diagnostic prompt construction, not a live scheduler admission.
/// IDs already belong to the model's tokenizer; this never invents text IDs.
/// A single request uses contiguous chunks and only the last produces a token.
pub fn encode_prompt(
  tokens : ArrayView[Int],
  limits : @protocol.WorkerProtocolLimits,
  inference : @inference.InferenceLimits,
  maximum_rows~ : Int,
  maximum_context~ : Int,
  model_generation~ : UInt64,
) -> Array[Bytes] raise {
  guard maximum_rows > 0 &&
    maximum_rows <= limits.max_plan_tokens &&
    maximum_context > 0 &&
    maximum_context <= limits.max_sequence_tokens &&
    !tokens.is_empty() &&
    tokens.length() < maximum_context &&
    model_generation > 0UL else {
    fail("invalid bounded serial prompt geometry")
  }
  let allocator = @pages.PageAllocator::new(
    @pages.PageAllocatorLimits::new(
      page_capacity=1,
      max_run_pages=1,
      max_active_references_per_page=1,
      max_cached_references_per_page=1,
    ),
  )
  let page = allocator.allocate_one()
  defer allocator.release_active(page)
  let results : Array[Bytes] = []
  let wire = @wire.PlanFrameBuffer::new(limits, inference)
  let scratch = FixedArray::make(wire.capacity(), b'\x00')
  let request_id = @inference.RequestId::new(1UL)
  let request_generation = @inference.RequestGeneration::initial()
  let capabilities = [@plan.KernelCapabilityId::paged_causal_attention()]
  for start = 0, sequence = 1UL; start < tokens.length(); {
    let end = (start + maximum_rows).min(tokens.length())
    let owner = @protocol.SchedulePlanBuffer::new(limits)
    let completion_slot = @protocol.CompletionSlot::new(0, limits)
    if end == tokens.length() {
      owner.append_final_prefill(
        request_id~,
        request_generation~,
        sequence_token_start=start,
        tokens=tokens[start:end],
        pages=[page],
        capabilities~,
        sampling=@inference.SamplingParameters::greedy(),
        sampling_seed=@inference.SamplingSeed::new(3UL),
        sample_index=end - 1,
        completion_slot~,
      )
    } else {
      owner.append_prefill(
        request_id~,
        request_generation~,
        sequence_token_start=start,
        tokens=tokens[start:end],
        pages=[page],
        capabilities~,
        completion_slot~,
      )
    }
    let submitted = owner.submit_wire(
      sequence_value=sequence,
      model_plan_generation_value=model_generation,
      token_budget=end - start,
      has_previous=sequence > 1UL,
      previous_sequence_value=sequence - 1UL,
      loaded_generation_value=model_generation,
    )
    let encoded = wire.encode_plan(submitted)
    encoded.copy_to(scratch, destination_offset=0)
    results.push(Bytes::from_array(scratch[:encoded.length()].to_owned()))
    ignore(owner.retire(submitted))
    continue end, sequence + 1UL
  }
  results
}

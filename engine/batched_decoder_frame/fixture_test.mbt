///|
fn limits() -> @protocol.WorkerProtocolLimits raise {
  @protocol.WorkerProtocolLimits::new(
    max_prefill_rows=4,
    max_decode_rows=4,
    max_plan_rows=4,
    max_plan_tokens=16,
    max_plan_pages=16,
    max_capabilities=8,
    max_completion_slots=4,
    max_sequence_tokens=32,
    max_token_id=31,
  )
}

///|
fn inference_limits() -> @inference.InferenceLimits raise {
  @inference.InferenceLimits::new(
    max_text_bytes=128,
    max_input_tokens=32,
    max_new_tokens=32,
    max_context_tokens=32,
    max_token_id=31,
    max_stop_token_ids=1,
    max_stop_strings=1,
    max_stop_string_bytes=1,
    max_trace_bytes=32,
    max_cache_scope_bytes=32,
    max_decoded_delta_bytes=32,
    max_deadline_millis=1000UL,
    max_top_k=8,
    max_temperature=4.0,
  )
}

///|
fn host() -> @batched_decoder_frame.BatchedDecoderFrame raise {
  @batched_decoder_frame.BatchedDecoderFrame::new(
    maximum_rows=8,
    maximum_sequences=3,
    cache_slots=3,
    maximum_context=8,
    vocabulary_size=16,
    model_generation=9UL,
  )
}

///|
// id, generation, position, tokens, decode, final-prefill, stochastic.
fn plan(
  entries : ArrayView[(UInt64, UInt64, Int, Array[Int], Bool, Bool, Bool)],
  model_generation? : UInt64 = 9UL,
) -> (
  @protocol.SubmittedSchedulePlan,
  @wire.ValidatedPlanFrame,
  @wire.CompletionFrameWriter,
) raise {
  let limits = limits()
  let inference = inference_limits()
  let owner = @protocol.SchedulePlanBuffer::new(limits)
  let allocator = @page_allocator.PageAllocator::new(
    @page_allocator.PageAllocatorLimits::new(
      page_capacity=16,
      max_run_pages=16,
      max_active_references_per_page=8,
      max_cached_references_per_page=8,
    ),
  )
  let mut total = 0
  for
    index, (id, generation, position, tokens, decode, final_prefill, stochastic) in entries {
    let request_id = @inference.RequestId::new(id)
    let request_generation = @inference.RequestGeneration::from_wire(generation)
    let completion_slot = @protocol.CompletionSlot::new(index, limits)
    let pages = [allocator.allocate_one()]
    let capabilities = [@plan.KernelCapabilityId::paged_causal_attention()]
    let sampling = if stochastic {
      @inference.SamplingParameters::sample(
        temperature=0.75,
        top_k=3,
        top_p=0.8,
        inference,
      )
    } else {
      @inference.SamplingParameters::greedy()
    }
    let sampling_seed = @inference.SamplingSeed::new(id + 1UL)
    if decode {
      owner.append_decode(
        request_id~,
        request_generation~,
        sequence_token_position=position,
        input_token=tokens[0],
        pages~,
        capabilities~,
        sampling~,
        sampling_seed~,
        sample_index=position,
        completion_slot~,
      )
      total += 1
    } else if final_prefill {
      owner.append_final_prefill(
        request_id~,
        request_generation~,
        sequence_token_start=position,
        tokens=tokens[:],
        pages~,
        capabilities~,
        sampling~,
        sampling_seed~,
        sample_index=position + tokens.length() - 1,
        completion_slot~,
      )
      total += tokens.length()
    } else {
      owner.append_prefill(
        request_id~,
        request_generation~,
        sequence_token_start=position,
        tokens=tokens[:],
        pages~,
        capabilities~,
        completion_slot~,
      )
      total += tokens.length()
    }
  }
  let submitted = owner.submit_wire(
    sequence_value=1UL,
    model_plan_generation_value=model_generation,
    token_budget=total,
    has_previous=false,
    previous_sequence_value=0UL,
    loaded_generation_value=model_generation,
  )
  let frame = @wire.PlanFrameBuffer::new(limits, inference).encode_plan(
    submitted,
  )
  let writer = @wire.CompletionFrameBuffer::new(limits, inference).writer_for(
    frame,
  )
  (submitted, frame, writer)
}

///|
fn port(
  host : @batched_decoder_frame.BatchedDecoderFrame,
  port : @batched_decoder_frame.BatchInputPort,
) -> Array[Int] raise {
  let bytes = FixedArray::make(host.input_capacity(port), b'\xff')
  let count = host.copy_input(port, bytes) / 4
  Array::makei(count, index => {
    for byte in 0..<4; bits = 0U {
      continue bits | (bytes[index * 4 + byte].to_uint() << (byte * 8))
    } nobreak {
      bits.reinterpret_as_int()
    }
  })
}

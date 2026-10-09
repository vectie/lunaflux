///|
fn limits() -> @protocol.WorkerProtocolLimits raise {
  @protocol.WorkerProtocolLimits::new(
    max_prefill_rows=2,
    max_decode_rows=2,
    max_plan_rows=4,
    max_plan_tokens=8,
    max_plan_pages=8,
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
fn host_frame() -> @serial_decoder_frame.SerialDecoderFrame raise {
  @serial_decoder_frame.SerialDecoderFrame::new(
    maximum_rows=4,
    maximum_context=12,
    vocabulary_size=16,
    model_generation=9UL,
  )
}

///|
fn request_plan(
  id? : UInt64 = 7UL,
  generation? : UInt64 = 0UL,
  position? : Int = 0,
  tokens? : ArrayView[Int] = [2, 3],
  decode? : Bool = false,
  produces? : Bool = false,
  stochastic? : Bool = false,
  model_generation? : UInt64 = 9UL,
) -> (
  @protocol.SubmittedSchedulePlan,
  @wire.ValidatedPlanFrame,
  @wire.CompletionFrameWriter,
) raise {
  let limits = limits()
  let inference_limits = inference_limits()
  let plan = @protocol.SchedulePlanBuffer::new(limits)
  let request_id = @inference.RequestId::new(id)
  let request_generation = @inference.RequestGeneration::from_wire(generation)
  let completion_slot = @protocol.CompletionSlot::new(0, limits)
  let sampling = if stochastic {
    @inference.SamplingParameters::sample(temperature=1.0, inference_limits)
  } else {
    @inference.SamplingParameters::greedy()
  }
  let sampling_seed = @inference.SamplingSeed::new(3UL)
  let allocator = @page_allocator.PageAllocator::new(
    @page_allocator.PageAllocatorLimits::new(
      page_capacity=8,
      max_run_pages=8,
      max_active_references_per_page=8,
      max_cached_references_per_page=8,
    ),
  )
  let pages = [allocator.allocate_one()]
  let capabilities = [@plan.KernelCapabilityId::paged_causal_attention()]
  if decode {
    plan.append_decode(
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
  } else if produces {
    plan.append_final_prefill(
      request_id~,
      request_generation~,
      sequence_token_start=position,
      tokens~,
      pages~,
      capabilities~,
      sampling~,
      sampling_seed~,
      sample_index=position + tokens.length() - 1,
      completion_slot~,
    )
  } else {
    plan.append_prefill(
      request_id~,
      request_generation~,
      sequence_token_start=position,
      tokens~,
      pages~,
      capabilities~,
      completion_slot~,
    )
  }
  let submitted = plan.submit_wire(
    sequence_value=1UL,
    model_plan_generation_value=model_generation,
    token_budget=if decode { 1 } else { tokens.length() },
    has_previous=false,
    previous_sequence_value=0UL,
    loaded_generation_value=model_generation,
  )
  let encoded = @wire.PlanFrameBuffer::new(limits, inference_limits).encode_plan(
    submitted,
  )
  let writer = @wire.CompletionFrameBuffer::new(limits, inference_limits).writer_for(
    encoded,
  )
  (submitted, encoded, writer)
}

///|
fn read_i32(bytes : FixedArray[Byte], index : Int) -> Int {
  for byte in 0..<4; bits = 0U {
    continue bits | (bytes[index * 4 + byte].to_uint() << (byte * 8))
  } nobreak {
    bits.reinterpret_as_int()
  }
}

///|
fn port_value(
  frame : @serial_decoder_frame.SerialDecoderFrame,
  port : Int,
  index? : Int = 0,
) -> Int raise {
  let bytes = FixedArray::make(frame.input_capacity(port), b'\x00')
  ignore(frame.copy_input(port, bytes))
  read_i32(bytes, index)
}

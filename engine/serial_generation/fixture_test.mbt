///|
fn limits() -> @protocol.WorkerProtocolLimits raise {
  @protocol.WorkerProtocolLimits::new(
    max_prefill_rows=1,
    max_decode_rows=1,
    max_plan_rows=2,
    max_plan_tokens=4,
    max_plan_pages=2,
    max_capabilities=3,
    max_completion_slots=2,
    max_sequence_tokens=16,
    max_token_id=31,
  )
}

///|
fn inference() -> @inference.InferenceLimits raise {
  @inference.InferenceLimits::new(
    max_text_bytes=128,
    max_input_tokens=16,
    max_new_tokens=16,
    max_context_tokens=16,
    max_token_id=31,
    max_stop_token_ids=2,
    max_stop_strings=1,
    max_stop_string_bytes=1,
    max_trace_bytes=16,
    max_cache_scope_bytes=16,
    max_decoded_delta_bytes=16,
    max_deadline_millis=1000UL,
    max_top_k=8,
    max_temperature=4.0,
  )
}

///|
fn seed(
  id? : UInt64 = 7UL,
  position? : Int = 0,
  produces? : Bool = true,
  sample? : Bool = false,
  sequence? : UInt64 = 1UL,
) -> @wire.ValidatedPlanFrame raise {
  let limits = limits()
  let allocator = @page_allocator.PageAllocator::new(
    @page_allocator.PageAllocatorLimits::new(
      page_capacity=2,
      max_run_pages=2,
      max_active_references_per_page=2,
      max_cached_references_per_page=2,
    ),
  )
  let owner = @protocol.SchedulePlanBuffer::new(limits)
  let pages = [allocator.allocate_one(), allocator.allocate_one()]
  let capabilities = [
    @plan.KernelCapabilityId::rms_norm(),
    @plan.KernelCapabilityId::positioned_rotary_embedding(),
    @plan.KernelCapabilityId::paged_causal_attention(),
  ]
  let request_id = @inference.RequestId::new(id)
  let request_generation = @inference.RequestGeneration::from_wire(3UL)
  let completion_slot = @protocol.CompletionSlot::new(1, limits)
  if produces {
    owner.append_final_prefill(
      request_id~,
      request_generation~,
      sequence_token_start=position,
      tokens=[2, 3],
      pages~,
      capabilities~,
      completion_slot~,
      sampling=if sample {
        @inference.SamplingParameters::sample(temperature=1.0, inference())
      } else {
        @inference.SamplingParameters::greedy()
      },
      sampling_seed=@inference.SamplingSeed::new(17UL),
      sample_index=position + 1,
    )
  } else {
    owner.append_prefill(
      request_id~,
      request_generation~,
      sequence_token_start=position,
      tokens=[2, 3],
      pages~,
      capabilities~,
      completion_slot~,
    )
  }
  let submitted = owner.submit_wire(
    sequence_value=sequence,
    model_plan_generation_value=9UL,
    token_budget=2,
    has_previous=sequence != 1UL,
    previous_sequence_value=sequence - 1UL,
    loaded_generation_value=9UL,
  )
  @wire.PlanFrameBuffer::new(limits, inference()).encode_plan(submitted)
}

///|
fn complete(
  plan : @wire.ValidatedPlanFrame,
  token : Int,
  failed? : Bool = false,
) -> @wire.ValidatedCompletionFrame raise {
  let writer = @wire.CompletionFrameBuffer::new(limits(), inference()).writer_for(
    plan,
  )
  if plan.prefill_count() == 1 {
    let row = plan.prefill_row(0)
    if failed {
      writer.append_prefill_failed(row, ExecutionFailed)
    } else if row.produces_token() {
      writer.append_final_prefill_completed(row, token)
    } else {
      writer.append_prefill_completed(row)
    }
  } else {
    let row = plan.decode_row(0)
    if failed {
      writer.append_decode_failed(row, ExecutionFailed)
    } else {
      writer.append_decode_completed(row, token)
    }
  }
  writer.submit()
}

///|
fn generation(
  plan : @wire.ValidatedPlanFrame,
  token : Int,
  maximum? : Int = 3,
  context? : Int = 16,
  stops? : ArrayView[Int] = [],
) -> @serial_generation.SerialGreedyGeneration raise {
  @serial_generation.SerialGreedyGeneration::new(
    plan,
    complete(plan, token),
    limits(),
    inference(),
    maximum_new_tokens=maximum,
    maximum_context=context,
    stop_tokens=stops,
  )
}

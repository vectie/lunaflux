///|
fn inference(v : Int32) -> @inference.InferenceLimits raise {
  @inference.InferenceLimits::new(
    max_text_bytes=128,
    max_input_tokens=16,
    max_new_tokens=16,
    max_context_tokens=16,
    max_token_id=v - 1,
    max_stop_token_ids=1,
    max_stop_strings=1,
    max_stop_string_bytes=1,
    max_trace_bytes=16,
    max_cache_scope_bytes=16,
    max_decoded_delta_bytes=16,
    max_deadline_millis=60000UL,
    max_top_k=v,
    max_temperature=4.0,
  )
}

///|
fn parameters(kind : Int32, v : Int32) -> @inference.SamplingParameters raise {
  match kind % 6 {
    0 => @inference.SamplingParameters::greedy()
    1 => @inference.SamplingParameters::sample(temperature=0.7, inference(v))
    2 =>
      @inference.SamplingParameters::sample(
        temperature=1.0,
        top_k=7,
        inference(v),
      )
    3 =>
      @inference.SamplingParameters::sample(
        temperature=1.5,
        top_p=0.8,
        inference(v),
      )
    4 =>
      @inference.SamplingParameters::sample(
        temperature=0.3,
        top_k=v,
        top_p=1.0,
        inference(v),
      )
    _ =>
      @inference.SamplingParameters::sample(
        temperature=1.0,
        top_k=1,
        top_p=0.01,
        inference(v),
      )
  }
}

///|
fn frame(
  v : Int32,
  rows : Int32,
  live : Int32,
  iteration : Int32,
) -> @wire.ValidatedPlanFrame raise {
  let limits = @protocol.WorkerProtocolLimits::new(
    max_prefill_rows=rows,
    max_decode_rows=rows,
    max_plan_rows=rows,
    max_plan_tokens=rows * 2,
    max_plan_pages=rows,
    max_capabilities=rows,
    max_completion_slots=rows,
    max_sequence_tokens=16,
    max_token_id=v - 1,
  )
  let owner = @protocol.SchedulePlanBuffer::new(limits)
  let allocator = @page_allocator.PageAllocator::new(
    @page_allocator.PageAllocatorLimits::new(
      page_capacity=rows,
      max_run_pages=rows,
      max_active_references_per_page=4,
      max_cached_references_per_page=4,
    ),
  )
  // Even a zero-output partial prefill uploads and retires its parameter lane.
  for row in 0..<live.max(1) {
    let pages = [allocator.allocate_one()]
    let capabilities = [@plan.KernelCapabilityId::paged_causal_attention()]
    if live == 0 {
      owner.append_prefill(
        request_id=@inference.RequestId::new(1UL),
        request_generation=@inference.RequestGeneration::initial(),
        sequence_token_start=0,
        tokens=[1],
        pages~,
        capabilities~,
        completion_slot=@protocol.CompletionSlot::new(0, limits),
      )
    } else {
      owner.append_final_prefill(
        request_id=@inference.RequestId::new(row.to_uint64() + 1UL),
        request_generation=@inference.RequestGeneration::initial(),
        sequence_token_start=0,
        tokens=[1],
        pages~,
        capabilities~,
        sampling=parameters(iteration + row, v),
        sampling_seed=@inference.SamplingSeed::new((row + 17).to_uint64()),
        sample_index=iteration,
        completion_slot=@protocol.CompletionSlot::new(row, limits),
      )
    }
  }
  let submitted = owner.submit_wire(
    sequence_value=iteration.to_uint64() + 1UL,
    model_plan_generation_value=1UL,
    token_budget=live.max(1),
    has_previous=false,
    previous_sequence_value=0UL,
    loaded_generation_value=1UL,
  )
  @wire.PlanFrameBuffer::new(limits, inference(v)).encode_plan(submitted)
}

///|
fn run_cases(
  context : @device.Context,
  logits : @device.Allocation,
  sampler : @gpu.BatchedDeviceSampler,
  v : Int32,
  rows : Int32,
) -> Unit raise {
  let payload = FixedArray::make(v * rows * 4, b'\x00')
  let values = Array::make(v, 0.0)
  let limits = @sampling.SamplingLimits::new(rows, v)
  let scratch = @sampling.SamplerScratch::new(v, limits)
  let expected = FixedArray::make(rows, -1)
  for iteration in 0..<24 {
    let live = if iteration == 7 {
      0
    } else if iteration % 3 == 0 {
      1
    } else {
      rows
    }
    for row in 0..<rows {
      for token in 0..<v {
        // Many ties, signed values and F32 increments lost by BF16 conversion.
        let x = if iteration == 8 {
          0.0F
        } else {
          (((token * 17 + row * 13) % 29).to_float() - 14.0F) * 0.03125F +
          (token % 3).to_float() * 0.00001F
        }
        values[token] = x.to_double()
        let bits = if iteration == 9 && token == v - 1 {
          0x7fc00000U
        } else {
          x.reinterpret_as_uint()
        }
        for b in 0..<4 {
          payload[(row * v + token) * 4 + b] = ((bits >> (b * 8)) & 255U).to_byte()
        }
      }
      let p = parameters(iteration + row, v)
      expected[row] = if iteration == 9 {
        -1
      } else if p.mode() == Greedy {
        @sampling.greedy(values[:], limits).token_id()
      } else {
        @sampling.stochastic_sample_at(
          values[:],
          p,
          limits,
          @inference.SamplingSeed::new((row + 17).to_uint64()),
          iteration,
          scratch,
        ).token_id()
      }
    }
    context.copy_from_fixed_host(
      logits,
      payload,
      0L,
      0L,
      payload.length().to_int64(),
    )
    let input = frame(v, rows, live, iteration)
    // A repetition must consume the same row seed/index rather than any global RNG.
    for _ in 0..<2 {
      sampler.prepare(input)
      sampler.begin()
      while !sampler.poll() {

      }
      for row in 0..<live {
        let actual = sampler.token_at(row)
        guard actual == expected[row] else {
          fail(
            "case=\{iteration} row=\{row} expected=\{expected[row]} actual=\{actual}",
          )
        }
      }
    }
  }
  println(
    "cases=24 repeats=2 full_f32=true unrestricted_k=true top_p=true nonfinite=true zero_output=true",
  )
}

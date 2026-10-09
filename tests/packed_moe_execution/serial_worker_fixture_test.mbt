///|
fn serial_limits() -> @protocol.WorkerProtocolLimits raise {
  @protocol.WorkerProtocolLimits::new(
    max_prefill_rows=1,
    max_decode_rows=1,
    max_plan_rows=2,
    max_plan_tokens=4,
    max_plan_pages=2,
    max_capabilities=2,
    max_completion_slots=2,
    max_sequence_tokens=16,
    max_token_id=31,
  )
}

///|
fn serial_inference_limits() -> @inference.InferenceLimits raise {
  @inference.InferenceLimits::new(
    max_text_bytes=128,
    max_input_tokens=16,
    max_new_tokens=16,
    max_context_tokens=16,
    max_token_id=31,
    max_stop_token_ids=1,
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
fn serial_wire_plan(
  id : UInt64,
  model_generation? : UInt64 = 9UL,
  position? : Int = 0,
  tokens? : ArrayView[Int] = [2, 3],
  decode? : Bool = false,
  produces? : Bool = true,
) -> (
  @protocol.SubmittedSchedulePlan,
  @wire.ValidatedPlanFrame,
  @wire.CompletionFrameWriter,
) raise {
  let limits = serial_limits()
  let inference = serial_inference_limits()
  let plan = @protocol.SchedulePlanBuffer::new(limits)
  let allocator = @page_allocator.PageAllocator::new(
    @page_allocator.PageAllocatorLimits::new(
      page_capacity=2,
      max_run_pages=2,
      max_active_references_per_page=2,
      max_cached_references_per_page=2,
    ),
  )
  let request_id = @inference.RequestId::new(id)
  let request_generation = @inference.RequestGeneration::initial()
  let pages = [allocator.allocate_one()]
  let capabilities = [@plan.KernelCapabilityId::paged_causal_attention()]
  let completion_slot = @protocol.CompletionSlot::new(0, limits)
  if decode {
    plan.append_decode(
      request_id~,
      request_generation~,
      sequence_token_position=position,
      input_token=tokens[0],
      pages~,
      capabilities~,
      sampling=@inference.SamplingParameters::greedy(),
      sampling_seed=@inference.SamplingSeed::new(3UL),
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
      sampling=@inference.SamplingParameters::greedy(),
      sampling_seed=@inference.SamplingSeed::new(3UL),
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
  let frame = @wire.PlanFrameBuffer::new(limits, inference).encode_plan(
    submitted,
  )
  let writer = @wire.CompletionFrameBuffer::new(limits, inference).writer_for(
    frame,
  )
  (submitted, frame, writer)
}

///|
fn serial_host() -> @serial_frame.SerialDecoderFrame raise {
  @serial_frame.SerialDecoderFrame::new(
    maximum_rows=4,
    maximum_context=16,
    vocabulary_size=16,
    model_generation=9UL,
  )
}

///|
fn serial_queue(
  context : @device.Context,
  aot : @device.Module,
  stream : @device.Stream,
  counts : @device.Allocation,
) -> (@device.OrderedKernelExecutor, @device.Function) raise {
  let function = aot.load_function(b"serial_fixture")
  let launch = @device.OrderedKernelLaunch::new(
    function,
    @device.KernelLaunchDimensions::new(
      grid_x=1,
      grid_y=1,
      grid_z=1,
      block_x=1,
      block_y=1,
      block_z=1,
      shared_memory_bytes=0,
    ),
    @device.KernelArguments::new([
      @device.KernelArgument::device_region(
        counts,
        offset=0L,
        byte_count=20L,
        alignment=4L,
      ),
    ]),
  )
  (context.create_ordered_kernel_executor(stream, [launch]), function)
}

///|
fn bind_serial_queue(
  worker : @serial_worker.SerialDecoderWorker,
  context : @device.Context,
  queue : @device.OrderedKernelExecutor,
  descriptors : ArrayView[@device.Allocation],
) -> Unit raise {
  worker.bind_executor(
    context,
    submit=rows => {
      assert_true(rows == 2)
      queue.enqueue(0)
      queue.record_completion()
    },
    poll=() => {
      if queue.poll_completion() {
        queue.reset()
        true
      } else {
        false
      }
    },
    close=() => {
      queue.abort()
      queue.close()
      // Match a model owner, whose layer teardown closes its error ports.
      for descriptor in descriptors {
        descriptor.close()
      }
    },
    append_descriptors=descriptors,
  )
}

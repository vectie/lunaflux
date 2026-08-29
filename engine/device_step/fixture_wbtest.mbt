///|
fn step_test_metadata() -> @spec.LlamaModelMetadata raise {
  @spec.LlamaModelMetadata::from_verified_content(
    @spec.LlamaModelSpec::new(
      vocabulary_size=128,
      hidden_size=16,
      intermediate_size=32,
      layer_count=2,
      attention_head_count=4,
      kv_head_count=2,
      context_length=64,
      rope_theta=10000.0,
      weight_dtype=BFloat16,
      kv_dtype=BFloat16,
    ),
    @spec.ContentDigest::from_sha256(
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
  )
}

///|
fn step_test_model_plan() -> @plan.ModelPlan raise {
  @llama.build_paged(step_test_metadata())
}

///|
fn step_test_identity() -> @spec.ModelIdentity raise {
  step_test_model_plan().identity()
}

///|
fn step_sampling_seed() -> @inference.SamplingSeed {
  try! @inference.SamplingSeed::new(0x123456789abcdef0UL)
}

///|
fn step_required_capabilities() -> FixedArray[@plan.KernelCapabilityId] {
  let source = (try! step_test_model_plan()).required_capabilities()
  let result = FixedArray::make(source.length(), source[0])
  for index, capability in source {
    result[index] = capability
  }
  result
}

///|
fn step_required_capability_array() -> Array[@plan.KernelCapabilityId] {
  let result : Array[@plan.KernelCapabilityId] = []
  for capability in (try! step_test_model_plan()).required_capabilities() {
    result.push(capability)
  }
  result
}

///|
fn step_test_layout(
  tokens_per_page? : Int = 4,
  pages? : Int = 64,
) -> @device_layout.DeviceKvLayout raise {
  @device_layout.plan(
    step_test_identity(),
    step_test_model_plan().kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page~,
      total_page_count=pages,
      max_arena_bytes=1048576L,
    ),
  )
}

///|
fn step_protocol_limits() -> @worker_protocol.WorkerProtocolLimits raise {
  @worker_protocol.WorkerProtocolLimits::new(
    max_prefill_rows=16,
    max_decode_rows=16,
    max_plan_rows=32,
    max_plan_tokens=128,
    max_plan_pages=128,
    max_capabilities=64,
    max_completion_slots=32,
    max_sequence_tokens=256,
    max_token_id=4096,
  )
}

///|
fn step_device_limits(
  max_rows? : Int = 4,
  max_tokens? : Int = 8,
  max_pages? : Int = 8,
  max_sequence_tokens? : Int = 256,
  max_token_id? : Int = 4096,
) -> DeviceStepLimits raise {
  DeviceStepLimits::new(
    max_rows~,
    max_tokens~,
    max_pages~,
    max_sequence_tokens~,
    max_token_id~,
  )
}

///|
fn step_allocator(capacity? : Int = 64) -> @page_allocator.PageAllocator raise {
  @page_allocator.PageAllocator::new(
    @page_allocator.PageAllocatorLimits::new(
      page_capacity=capacity,
      max_run_pages=capacity,
      max_active_references_per_page=8,
      max_cached_references_per_page=8,
    ),
  )
}

///|
fn step_slot(value : Int) -> @worker_protocol.CompletionSlot raise {
  @worker_protocol.CompletionSlot::new(value, step_protocol_limits())
}

///|
fn step_request(value : UInt64) -> @inference.RequestId raise {
  @inference.RequestId::new(value)
}

///|
fn submit_step_buffer(
  buffer : @worker_protocol.SchedulePlanBuffer,
  sequence? : UInt64 = 1UL,
  generation? : UInt64 = 1UL,
  token_budget~ : Int,
) -> @worker_protocol.SubmittedSchedulePlan raise {
  let sequence_id = @worker_protocol.PlanSequence::from_wire(sequence)
  let generation_id = @worker_protocol.ModelPlanGeneration::from_wire(
    generation,
  )
  buffer.submit(
    sequence=sequence_id,
    model_plan_generation=generation_id,
    token_budget~,
    if sequence == 1UL {
      None
    } else {
      Some(@worker_protocol.PlanSequence::from_wire(sequence - 1UL))
    },
    generation_id,
  )
}

///|
fn read_host_i32(buffer : FixedArray[Byte], index : Int) -> Int {
  let start = index * INT32_BYTES
  buffer[start].to_int() |
  (buffer[start + 1].to_int() << 8) |
  (buffer[start + 2].to_int() << 16) |
  (buffer[start + 3].to_int() << 24)
}

///|
fn preflight_then_write(
  plan : @worker_protocol.SubmittedSchedulePlan,
  expected_generation : UInt64,
  expected_identity : @spec.ModelIdentity,
  layout : @device_layout.DeviceKvLayout,
  limits : DeviceStepLimits,
  host : DeviceStepHostBuffers,
) -> DeviceStepSummary raise DeviceStepError {
  preflight_and_write(
    plan,
    expected_generation,
    expected_identity,
    layout,
    limits,
    host,
    step_required_capabilities(),
    true,
  )
}

///|
extern "C" fn alloc_probe_begin() = "lunaflux_device_step_alloc_probe_begin"

///|
extern "C" fn alloc_probe_end() -> UInt64 = "lunaflux_device_step_alloc_probe_end"

///|
extern "C" fn alloc_probe_check(observed : UInt64, expect_zero : Int) -> Int = "lunaflux_device_step_alloc_probe_check"

///|
extern "C" fn alloc_probe_abort() = "lunaflux_device_step_alloc_probe_abort"

///|
extern "C" fn alloc_probe_seed() -> Int = "lunaflux_device_step_alloc_probe_seed"

///|
#borrow(value)
extern "C" fn keep_array_alive(value : FixedArray[Int]) = "lunaflux_device_step_alloc_probe_keep_alive"

///|
extern "C" fn fake_copy_count() -> UInt64 = "lunaflux_device_step_fake_copy_count"

///|
struct PositiveControl {
  value : Int
}

///|
let positive_control_sink : FixedArray[PositiveControl?] = FixedArray::make(
  1,
  None,
)

///|
fn verify_probe_observes_allocations() -> Unit {
  alloc_probe_begin()
  let record = { value: alloc_probe_seed() }
  positive_control_sink[0] = Some(record)
  let record_observed = alloc_probe_end()
  positive_control_sink[0] = None
  if alloc_probe_check(record_observed, 0) != 1 {
    alloc_probe_abort()
  }

  alloc_probe_begin()
  let array = FixedArray::make(8, record.value)
  keep_array_alive(array)
  let array_observed = alloc_probe_end()
  if alloc_probe_check(array_observed, 0) != 1 {
    alloc_probe_abort()
  }
}

///|
fn metadata() -> @spec.LlamaModelMetadata {
  try! @spec.LlamaModelMetadata::from_verified_content(
    try! @spec.LlamaModelSpec::new(
      vocabulary_size=128,
      hidden_size=16,
      intermediate_size=32,
      layer_count=2,
      attention_head_count=4,
      kv_head_count=2,
      context_length=16,
      rope_theta=10000.0,
      weight_dtype=BFloat16,
      kv_dtype=BFloat16,
    ),
    try! @spec.ContentDigest::from_sha256(
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
  )
}

///|
fn worker_limits() -> @worker_protocol.WorkerProtocolLimits {
  try! @worker_protocol.WorkerProtocolLimits::new(
    max_prefill_rows=1,
    max_decode_rows=1,
    max_plan_rows=1,
    max_plan_tokens=2,
    max_plan_pages=1,
    max_capabilities=64,
    max_completion_slots=1,
    max_sequence_tokens=16,
    max_token_id=127,
  )
}

///|
fn device_limits() -> @device_step.DeviceStepLimits {
  try! @device_step.DeviceStepLimits::new(
    max_rows=1,
    max_tokens=2,
    max_pages=1,
    max_sequence_tokens=16,
    max_token_id=127,
  )
}

///|
fn page_id() -> @page_allocator.PageId {
  let allocator = @page_allocator.PageAllocator::new(
    try! @page_allocator.PageAllocatorLimits::new(
      page_capacity=1,
      max_run_pages=1,
      max_active_references_per_page=1,
      max_cached_references_per_page=1,
    ),
  )
  try! allocator.allocate_one()
}

///|
fn submitted_plan(
  sequence_value : UInt64,
  model_plan : @plan.ModelPlan,
  page : @page_allocator.PageId,
) -> @worker_protocol.SubmittedSchedulePlan {
  let limits = worker_limits()
  let buffer = try! @worker_protocol.SchedulePlanBuffer::new(limits)
  try! buffer.append_prefill(
    request_id=try! @inference.RequestId::new(1UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_start=0,
    tokens=[10, 11],
    pages=[page],
    capabilities=model_plan.required_capabilities(),
    completion_slot=try! @worker_protocol.CompletionSlot::new(0, limits),
  )
  let generation = @worker_protocol.ModelPlanGeneration::first()
  try! buffer.submit(
    sequence=try! @worker_protocol.PlanSequence::from_wire(sequence_value),
    model_plan_generation=generation,
    token_budget=2,
    if sequence_value == 1UL {
      None
    } else {
      Some(try! @worker_protocol.PlanSequence::from_wire(sequence_value - 1UL))
    },
    generation,
  )
}

///|
fn prepared_plans(
  count : Int,
  model_plan : @plan.ModelPlan,
) -> FixedArray[@worker_protocol.SubmittedSchedulePlan] {
  let page = page_id()
  let first = submitted_plan(1UL, model_plan, page)
  let plans = FixedArray::make(count, first)
  for index in 1..<count {
    plans[index] = submitted_plan((index + 1).to_uint64(), model_plan, page)
  }
  plans
}

///|
fn prepared_frames(
  plans : FixedArray[@worker_protocol.SubmittedSchedulePlan],
) -> FixedArray[@worker_wire.ValidatedPlanFrame] {
  let first_owner = try! @worker_wire.PlanFrameBuffer::new(
    worker_limits(),
    try! @inference.InferenceLimits::new(
      max_text_bytes=64,
      max_input_tokens=16,
      max_new_tokens=16,
      max_context_tokens=16,
      max_token_id=127,
      max_stop_token_ids=1,
      max_stop_strings=1,
      max_stop_string_bytes=1,
      max_trace_bytes=1,
      max_cache_scope_bytes=1,
      max_decoded_delta_bytes=1,
      max_deadline_millis=100UL,
      max_top_k=128,
      max_temperature=4.0,
    ),
  )
  let first = try! first_owner.encode_plan(plans[0])
  let frames = FixedArray::make(plans.length(), first)
  for index in 1..<plans.length() {
    let owner = try! @worker_wire.PlanFrameBuffer::new(
      worker_limits(),
      try! @inference.InferenceLimits::new(
        max_text_bytes=64,
        max_input_tokens=16,
        max_new_tokens=16,
        max_context_tokens=16,
        max_token_id=127,
        max_stop_token_ids=1,
        max_stop_strings=1,
        max_stop_string_bytes=1,
        max_trace_bytes=1,
        max_cache_scope_bytes=1,
        max_decoded_delta_bytes=1,
        max_deadline_millis=100UL,
        max_top_k=128,
        max_temperature=4.0,
      ),
    )
    frames[index] = try! owner.encode_plan(plans[index])
  }
  frames
}

///|
fn main {
  verify_probe_observes_allocations()
  let model_plan = try! @llama.build_paged(metadata())
  let layout = try! @device_layout.plan(
    model_plan.identity(),
    model_plan.kv_cache_geometry(),
    try! @device_layout.DeviceKvLimits::new(
      tokens_per_page=4,
      total_page_count=1,
      max_arena_bytes=1048576L,
    ),
  )
  let cycle_count = 128
  let plans = prepared_plans(cycle_count + 1, model_plan)
  let frames = prepared_frames(plans)
  let context = try! @device.open_context(0)
  let owner = match
    (try! @device_step.prepare(
      context,
      model_plan,
      @worker_protocol.ModelPlanGeneration::first(),
      layout,
      device_limits(),
      has_accepted_predecessor=false,
      accepted_predecessor_value=0UL,
    )) {
    Ready(owner) => owner
    CleanupRequired(_) => {
      alloc_probe_abort()
      abort("unreachable")
    }
  }

  let warm = try! owner.stage_frame(frames[0])
  try! owner.finish(warm)
  let copies_before = fake_copy_count()

  alloc_probe_begin()
  for index in 1..<=cycle_count {
    let staged = try! owner.stage_frame(frames[index])
    try! owner.finish(staged)
  }
  let observed = alloc_probe_end()
  let copies_after = fake_copy_count()
  if alloc_probe_check(observed, 1) != 1 ||
    copies_after - copies_before != cycle_count.to_uint64() * 7UL {
    alloc_probe_abort()
  }

  try! owner.close()
  try! context.close()
}

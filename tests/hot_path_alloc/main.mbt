///|
extern "C" fn alloc_probe_begin() = "lunaflux_alloc_probe_begin"

///|
extern "C" fn alloc_probe_end() -> UInt64 = "lunaflux_alloc_probe_end"

///|
extern "C" fn alloc_probe_check(observed : UInt64, expect_zero : Int) -> Int = "lunaflux_alloc_probe_check"

///|
extern "C" fn alloc_probe_abort() = "lunaflux_alloc_probe_abort"

///|
extern "C" fn alloc_probe_seed() -> Int = "lunaflux_alloc_probe_seed"

///|
#borrow(value)
extern "C" fn keep_array_alive(value : FixedArray[Int]) = "lunaflux_alloc_probe_keep_alive"

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
  // Both generated-C record allocation and runtime array allocation must be
  // observable, or a zero result below would not be trustworthy.
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
fn model_identity() -> @spec.ModelIdentity {
  let content = try! @spec.ContentDigest::from_sha256(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  let plan = try! @spec.PlanDigest::from_sha256(
    "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
  )
  @spec.ModelIdentity::new(content, plan)
}

///|
fn inference_limits() -> @inference.InferenceLimits {
  try! @inference.InferenceLimits::new(
    max_text_bytes=64,
    max_input_tokens=8,
    max_new_tokens=24,
    max_context_tokens=32,
    max_token_id=63,
    max_stop_token_ids=4,
    max_stop_strings=2,
    max_stop_string_bytes=8,
    max_trace_bytes=8,
    max_cache_scope_bytes=8,
    max_decoded_delta_bytes=8,
    max_deadline_millis=1000UL,
    max_top_k=64,
    max_temperature=4.0,
  )
}

///|
fn worker_limits() -> @worker_protocol.WorkerProtocolLimits {
  try! @worker_protocol.WorkerProtocolLimits::new(
    max_prefill_rows=1,
    max_decode_rows=1,
    max_plan_rows=1,
    max_plan_tokens=1,
    max_plan_pages=8,
    max_capabilities=4,
    max_completion_slots=1,
    max_sequence_tokens=32,
    max_token_id=63,
  )
}

///|
fn resolved_capacity() -> @runtime_resolved.ResolvedRuntimeCapacity {
  let scheduler = try! @scheduler_config.new(
    step_token_budget=1,
    max_active_requests=1,
    max_waiting_requests=1,
    prefill_chunk_tokens=1,
    emergency_decode_page_reserve=1,
    waiting_age_threshold_steps=2,
    output_event_capacity=24,
  )
  let cache = try! @cache.new(
    tokens_per_page=4,
    total_page_count=10,
    block_table_pages_per_request=8,
    prefix_enabled=false,
    max_prefix_entries=0,
    max_prefix_nodes=0,
    max_prefix_tokens_per_entry=0,
    max_prefix_pages=0,
    max_active_references_per_page=2,
    max_cached_references_per_page=2,
    layout_version=1,
  )
  let shapes = try! @plan.ShapeConstraints::new(
    max_batch_rows=1,
    max_sequence_tokens=32,
    vocabulary_size=64,
    hidden_size=8,
    intermediate_size=16,
    attention_head_count=2,
    kv_head_count=1,
    head_dimension=4,
  )
  try! @runtime_resolved.resolve(scheduler, cache, shapes, worker_limits())
}

///|
fn scheduler() -> @scheduler.Scheduler {
  let identity = model_identity()
  let generation = @worker_protocol.ModelPlanGeneration::first()
  let recipe = try! @worker_protocol.RowCapabilityRecipe::new(
    identity~,
    model_plan_generation=generation,
    [@plan.KernelCapabilityId::embedding_lookup()],
    worker_limits(),
  )
  try! @scheduler.Scheduler::new(
    resolved_capacity(),
    identity,
    generation,
    recipe,
    recipe,
    recipe,
  )
}

///|
fn request() -> @scheduler.TokenizedRequest {
  let limits = inference_limits()
  try! @scheduler.TokenizedRequest::new(
    request_id=try! @inference.RequestId::new(1UL),
    model_identity=model_identity(),
    input_tokens=try! @inference.TokenBuffer::new([1], limits),
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=try! @inference.SamplingSeed::new(1UL),
    stops=try! @inference.StopConditions::new([], [], limits),
    cache=@inference.CachePolicy::new(
      try! @inference.CacheScope::from_ascii(b"alloc", limits),
      Disabled,
    ),
    stream_preference=Streaming,
    effective_limits=try! @inference.EffectiveLimits::new(
      max_new_tokens=24,
      context_ceiling=32,
      limits,
    ),
    deadline_budget=try! @inference.DeadlineBudget::from_milliseconds(
      1000UL, limits,
    ),
  )
}

///|
fn complete_final_prefill(owner : @scheduler.Scheduler) -> Unit {
  let outcome = try! owner.build_next()
  let submitted = try! owner.submitted_plan(outcome)
  let row = try! submitted.prefill_row(0)
  let writer = try! owner.completion_writer(submitted)
  try! writer.append_final_prefill_completed(
    slot=try! row.completion_slot(),
    request_id=try! row.request_id(),
    request_generation=try! row.request_generation(),
    processed_tokens=(try! row.tokens()).length(),
    token_id=2,
  )
  try! owner.complete(submitted, try! writer.submit())
}

///|
fn complete_decode(owner : @scheduler.Scheduler, token : Int) -> Unit {
  let outcome = try! owner.build_next()
  let submitted = try! owner.submitted_plan(outcome)
  let row = try! submitted.decode_row(0)
  let writer = try! owner.completion_writer(submitted)
  try! writer.append_decode_completed(
    slot=try! row.completion_slot(),
    request_id=try! row.request_id(),
    request_generation=try! row.request_generation(),
    token_id=token,
  )
  try! owner.complete(submitted, try! writer.submit())
}

///|
fn main {
  verify_probe_observes_allocations()
  let owner = scheduler()
  ignore(try! owner.admit(request(), 0UL))

  // Warm both scheduler-owned A/B plan and completion buffers before counting.
  complete_final_prefill(owner)
  complete_decode(owner, 3)
  let pages_before = owner.free_page_count()

  // Decode enough tokens to reuse both A/B protocol buffers and cross at
  // least one KV page boundary while the counter is active.
  alloc_probe_begin()
  for token = 4; token < 12; token = token + 1 {
    complete_decode(owner, token)
  }
  let observed = alloc_probe_end()
  let pages_after = owner.free_page_count()
  if alloc_probe_check(observed, 1) != 1 {
    alloc_probe_abort()
  }
  if pages_after >= pages_before {
    alloc_probe_abort()
  }
}

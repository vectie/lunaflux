///|
fn test_identity(seed : String) -> @spec.ModelIdentity {
  let content = try! @spec.ContentDigest::from_sha256(seed)
  let plan = try! @spec.PlanDigest::from_sha256(
    "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
  )
  try! @spec.ModelIdentity::new(content, plan)
}

///|
fn selected_identity() -> @spec.ModelIdentity {
  test_identity(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
}

///|
fn inference_limits() -> @inference.InferenceLimits {
  try! @inference.InferenceLimits::new(
    max_text_bytes=64,
    max_input_tokens=8,
    max_new_tokens=4,
    max_context_tokens=16,
    max_token_id=31,
    max_stop_token_ids=4,
    max_stop_strings=2,
    max_stop_string_bytes=8,
    max_trace_bytes=8,
    max_cache_scope_bytes=8,
    max_decoded_delta_bytes=8,
    max_deadline_millis=1000UL,
    max_top_k=32,
    max_temperature=4.0,
  )
}

///|
fn stop_token_projection(
  limits : @inference.InferenceLimits,
  token_ids : ArrayView[Int],
  stop_strings? : ArrayView[Bytes] = [],
) -> @inference.LunaRequestStopTokenView {
  let storage = try! @inference.LunaRequestSemanticStorage::new(
    limits,
    try! @inference.LunaRequestSemanticStepBudget::new(17),
  )
  let write = try! storage.begin(
    stop_token_count=token_ids.length(),
    stop_string_count=stop_strings.length(),
    cache_scope_byte_count=5,
    cache_permission=Disabled,
  )
  for token in token_ids {
    try! write.push_stop_token(token)
  }
  for string in stop_strings {
    try! write.push_stop_string_length(string.length())
    for byte in string {
      try! write.push_stop_string_byte(byte)
    }
  }
  for byte in b"scope" {
    try! write.push_cache_scope_byte(byte)
  }
  let work = try! write.finish()
  while (try! work.progress()).is_pending() {

  }
  try! (try! work.take_lease()).stop_token_view()
}

///|
fn resolved_capacity() -> @runtime_resolved.ResolvedRuntimeCapacity {
  let scheduler = try! @scheduler.new(
    step_token_budget=4,
    max_active_requests=2,
    max_waiting_requests=2,
    prefill_chunk_tokens=2,
    emergency_decode_page_reserve=1,
    waiting_age_threshold_steps=2,
    output_event_capacity=4,
  )
  let cache = try! @cache.new(
    tokens_per_page=4,
    total_page_count=8,
    block_table_pages_per_request=4,
    prefix_enabled=false,
    max_prefix_entries=0,
    max_prefix_nodes=0,
    max_prefix_tokens_per_entry=0,
    max_prefix_pages=0,
    max_active_references_per_page=4,
    max_cached_references_per_page=4,
    layout_version=1,
  )
  let shapes = try! @plan.ShapeConstraints::new(
    max_batch_rows=2,
    max_sequence_tokens=16,
    vocabulary_size=32,
    hidden_size=8,
    intermediate_size=16,
    attention_head_count=2,
    kv_head_count=1,
    head_dimension=4,
  )
  let worker = try! @worker_protocol.WorkerProtocolLimits::new(
    max_prefill_rows=2,
    max_decode_rows=2,
    max_plan_rows=2,
    max_plan_tokens=4,
    max_plan_pages=8,
    max_capabilities=8,
    max_completion_slots=2,
    max_sequence_tokens=16,
    max_token_id=31,
  )
  try! @runtime_resolved.resolve(scheduler, cache, shapes, worker)
}

///|
fn tight_resolved_capacity() -> @runtime_resolved.ResolvedRuntimeCapacity {
  let scheduler = try! @scheduler.new(
    step_token_budget=2,
    max_active_requests=1,
    max_waiting_requests=1,
    prefill_chunk_tokens=2,
    emergency_decode_page_reserve=2,
    waiting_age_threshold_steps=2,
    output_event_capacity=2,
  )
  let cache = try! @cache.new(
    tokens_per_page=4,
    total_page_count=3,
    block_table_pages_per_request=3,
    prefix_enabled=false,
    max_prefix_entries=0,
    max_prefix_nodes=0,
    max_prefix_tokens_per_entry=0,
    max_prefix_pages=0,
    max_active_references_per_page=4,
    max_cached_references_per_page=4,
    layout_version=1,
  )
  let shapes = try! @plan.ShapeConstraints::new(
    max_batch_rows=1,
    max_sequence_tokens=12,
    vocabulary_size=32,
    hidden_size=8,
    intermediate_size=16,
    attention_head_count=2,
    kv_head_count=1,
    head_dimension=4,
  )
  let worker = try! @worker_protocol.WorkerProtocolLimits::new(
    max_prefill_rows=1,
    max_decode_rows=1,
    max_plan_rows=1,
    max_plan_tokens=2,
    max_plan_pages=3,
    max_capabilities=8,
    max_completion_slots=1,
    max_sequence_tokens=12,
    max_token_id=31,
  )
  try! @runtime_resolved.resolve(scheduler, cache, shapes, worker)
}

///|
fn request_with_tokens(
  id : UInt64,
  tokens : ArrayView[Int],
  max_new_tokens : Int,
  context_ceiling : Int,
) -> TokenizedRequest {
  let limits = inference_limits()
  try! TokenizedRequest::new(
    request_id=try! @inference.RequestId::new(id),
    model_identity=selected_identity(),
    input_tokens=try! @inference.TokenBuffer::new(tokens, limits),
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=try! @inference.SamplingSeed::new(id),
    stop_tokens=stop_token_projection(limits, []),
    stream_preference=Streaming,
    effective_limits=try! @inference.EffectiveLimits::new(
      max_new_tokens~,
      context_ceiling~,
      limits,
    ),
    admission_deadline=try! @inference.AdmissionDeadline::from_receipt(
      0UL,
      try! @inference.DeadlineBudget::from_milliseconds(100UL, limits),
    ),
  )
}

///|
fn request(
  id : UInt64,
  identity? : @spec.ModelIdentity = selected_identity(),
  deadline? : UInt64 = 100UL,
  max_new_tokens? : Int = 2,
  context_ceiling? : Int = 8,
) -> TokenizedRequest {
  let limits = inference_limits()
  let input = try! @inference.TokenBuffer::new([1, 2], limits)
  try! TokenizedRequest::new(
    request_id=try! @inference.RequestId::new(id),
    model_identity=identity,
    input_tokens=input,
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=try! @inference.SamplingSeed::new(id),
    stop_tokens=stop_token_projection(limits, []),
    stream_preference=Streaming,
    effective_limits=try! @inference.EffectiveLimits::new(
      max_new_tokens~,
      context_ceiling~,
      limits,
    ),
    admission_deadline=try! @inference.AdmissionDeadline::from_receipt(
      0UL,
      try! @inference.DeadlineBudget::from_milliseconds(deadline, limits),
    ),
  )
}

///|
fn scheduler() -> Scheduler {
  let limits = worker_limits()
  let generation = @worker_protocol.ModelPlanGeneration::first()
  let recipe = try! @worker_protocol.RowCapabilityRecipe::new(
    identity=selected_identity(),
    model_plan_generation=generation,
    [@plan.KernelCapabilityId::embedding_lookup()],
    limits,
  )
  try! Scheduler::new(
    resolved_capacity(),
    selected_identity(),
    generation,
    recipe,
    recipe,
    recipe,
  )
}

///|
fn worker_limits() -> @worker_protocol.WorkerProtocolLimits {
  try! @worker_protocol.WorkerProtocolLimits::new(
    max_prefill_rows=2,
    max_decode_rows=2,
    max_plan_rows=2,
    max_plan_tokens=4,
    max_plan_pages=8,
    max_capabilities=8,
    max_completion_slots=2,
    max_sequence_tokens=16,
    max_token_id=31,
  )
}

///|
fn tight_scheduler() -> Scheduler {
  let limits = worker_limits()
  let generation = @worker_protocol.ModelPlanGeneration::first()
  let recipe = try! @worker_protocol.RowCapabilityRecipe::new(
    identity=selected_identity(),
    model_plan_generation=generation,
    [@plan.KernelCapabilityId::embedding_lookup()],
    limits,
  )
  try! Scheduler::new(
    tight_resolved_capacity(),
    selected_identity(),
    generation,
    recipe,
    recipe,
    recipe,
  )
}

///|
fn next_plan(owner : Scheduler) -> @worker_protocol.SubmittedSchedulePlan {
  try! owner.submitted_plan(try! owner.build_next())
}

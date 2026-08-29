///|
fn wb_prefix_identity() -> @spec.ModelIdentity {
  try! @spec.ModelIdentity::new(
    try! @spec.ContentDigest::from_sha256(
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
    try! @spec.PlanDigest::from_sha256(
      "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
    ),
  )
}

///|
fn wb_prefix_inference_limits() -> @inference.InferenceLimits {
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
    max_cache_scope_bytes=16,
    max_decoded_delta_bytes=8,
    max_deadline_millis=1000UL,
    max_top_k=32,
    max_temperature=4.0,
  )
}

///|
fn wb_prefix_worker_limits() -> @worker_protocol.WorkerProtocolLimits {
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
fn wb_prefix_capacity() -> @runtime_resolved.ResolvedRuntimeCapacity {
  let scheduler = try! @scheduler.new(
    step_token_budget=4,
    max_active_requests=2,
    max_waiting_requests=2,
    prefill_chunk_tokens=4,
    emergency_decode_page_reserve=1,
    waiting_age_threshold_steps=2,
    output_event_capacity=4,
  )
  let cache = try! @cache.new(
    tokens_per_page=2,
    total_page_count=16,
    block_table_pages_per_request=8,
    prefix_enabled=true,
    max_prefix_entries=4,
    max_prefix_nodes=32,
    max_prefix_tokens_per_entry=8,
    max_prefix_pages=16,
    max_prefix_scope_bytes=8,
    max_active_references_per_page=4,
    max_cached_references_per_page=1,
    layout_version=3,
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
    max_plan_pages=16,
    max_capabilities=8,
    max_completion_slots=2,
    max_sequence_tokens=16,
    max_token_id=31,
  )
  try! @runtime_resolved.resolve(scheduler, cache, shapes, worker)
}

///|
fn scheduler_prefix_allocation_positive_control() -> FixedArray[Byte] {
  FixedArray::make(23, b'P')
}

///|
test "scheduler prefix allocation positive control is linked" {
  assert_eq(scheduler_prefix_allocation_positive_control().length(), 23)
}

///|
fn wb_prefix_scheduler() -> Scheduler {
  let limits = wb_prefix_worker_limits()
  let generation = @worker_protocol.ModelPlanGeneration::first()
  let recipe = try! @worker_protocol.RowCapabilityRecipe::new(
    identity=wb_prefix_identity(),
    model_plan_generation=generation,
    [@plan.KernelCapabilityId::embedding_lookup()],
    limits,
  )
  try! Scheduler::new(
    wb_prefix_capacity(),
    wb_prefix_identity(),
    generation,
    recipe,
    recipe,
    recipe,
  )
}

///|
fn wb_prefix_request(
  id : UInt64,
  tokens : ArrayView[Int],
  permission : @inference.CachePermission,
  scope : Bytes,
  max_new_tokens? : Int = 1,
) -> (TokenizedRequest, @inference.LunaRequestSemanticLease) {
  let limits = wb_prefix_inference_limits()
  let storage = try! @inference.LunaRequestSemanticStorage::new(
    limits,
    try! @inference.LunaRequestSemanticStepBudget::new(17),
  )
  let write = try! storage.begin(
    stop_token_count=0,
    stop_string_count=0,
    cache_scope_byte_count=scope.length(),
    cache_permission=permission,
  )
  for byte in scope {
    try! write.push_cache_scope_byte(byte)
  }
  let work = try! write.finish()
  while (try! work.progress()).is_pending() {

  }
  let lease = try! work.take_lease()
  let request = try! TokenizedRequest::new_with_prefix(
    request_id=try! @inference.RequestId::new(id),
    model_identity=wb_prefix_identity(),
    input_tokens=try! @inference.TokenBuffer::new(tokens, limits),
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=try! @inference.SamplingSeed::new(id),
    stop_tokens=try! lease.stop_token_view(),
    stream_preference=Streaming,
    effective_limits=try! @inference.EffectiveLimits::new(
      max_new_tokens~,
      context_ceiling=8,
      limits,
    ),
    admission_deadline=try! @inference.AdmissionDeadline::from_receipt(
      0UL,
      try! @inference.DeadlineBudget::from_milliseconds(100UL, limits),
    ),
    tokenizer_digest=try! @tokenizer.TokenizerDigest::parse(
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
    prefix_semantic=try! lease.prefix_view(),
  )
  (request, lease)
}

///|
fn wb_complete_one_token_request(owner : Scheduler, sampled : Int) -> Unit {
  let first = wb_next_plan(owner)
  if (try! first.prefill_row(0).tokens().length()) == 4 {
    ignore(try! owner.complete(first, wb_prefill_completion(owner, first)))
  } else {
    ignore(
      try! owner.complete(first, wb_final_completion(owner, first, sampled)),
    )
    return
  }
  let final_plan = wb_next_plan(owner)
  ignore(
    try! owner.complete(
      final_plan,
      wb_final_completion(owner, final_plan, sampled),
    ),
  )
}

///|
test "cached and uncached host requests preserve semantics and processed count" {
  let owner = wb_prefix_scheduler()
  let (first, first_lease) = wb_prefix_request(
    901UL,
    [1, 2, 3, 4, 5],
    ReadWrite,
    b"scope",
  )
  ignore(try! owner.admit(first, 0UL))
  wb_complete_one_token_request(owner, 7)
  assert_eq((try! owner.take_generated_token()).token_id(), 7)
  let first_terminal = try! owner.take_terminal_notice()
  assert_eq(first_terminal.reason(), MaximumOutputReached)
  assert_eq(first_terminal.cached_input_tokens(), 0)
  try! first_lease.release()
  let after_publish = owner.luna_telemetry()
  assert_eq(after_publish.prefix_publications(), 1UL)
  assert_eq(after_publish.prefix_entries(), 1)
  assert_eq(after_publish.prefix_pages(), 2)
  assert_eq(after_publish.prefix_tokens_computed(), 5UL)

  let (cached, cached_lease) = wb_prefix_request(
    902UL,
    [1, 2, 3, 4, 9],
    ReadOnly,
    b"scope",
  )
  let cached_handle = try! owner.admit(cached, 0UL)
  let cached_status = owner.status(cached_handle.request_id()).unwrap()
  assert_eq(cached_status.processed_input_tokens(), 0)
  let hit = owner.luna_telemetry()
  assert_eq(hit.prefix_lookups(), 2UL)
  assert_eq(hit.prefix_hits(), 0UL)
  assert_eq(hit.prefix_tokens_reused(), 0UL)
  let cached_plan = wb_next_plan(owner)
  assert_eq(owner.luna_telemetry().prefix_hits(), 1UL)
  assert_eq(owner.luna_telemetry().prefix_tokens_reused(), 4UL)
  assert_eq(
    owner.status(cached_handle.request_id()).unwrap().cached_input_tokens(),
    4,
  )
  assert_eq(try! cached_plan.prefill_row(0).sequence_token_start(), 4)
  assert_eq(try! cached_plan.prefill_row(0).tokens().length(), 1)
  assert_eq(try! cached_plan.token_at(0), 9)
  ignore(
    try! owner.complete(cached_plan, wb_final_completion(owner, cached_plan, 7)),
  )
  assert_eq((try! owner.take_generated_token()).token_id(), 7)
  let cached_terminal = try! owner.take_terminal_notice()
  assert_eq(cached_terminal.cached_input_tokens(), 4)
  try! cached_lease.release()
  let final_telemetry = owner.luna_telemetry()
  assert_eq(final_telemetry.prefix_tokens_computed(), 6UL)
  assert_eq(final_telemetry.prefix_entries(), 1)
  assert_eq(final_telemetry.prefix_pages(), 2)
  assert_eq(final_telemetry.kv_pages_used(), 2)
}

///|
test "scope mismatch and disabled permission fail closed without false hits" {
  let owner = wb_prefix_scheduler()
  let (seed, seed_lease) = wb_prefix_request(
    911UL,
    [1, 2, 3, 4, 5],
    ReadWrite,
    b"scope",
  )
  ignore(try! owner.admit(seed, 0UL))
  wb_complete_one_token_request(owner, 7)
  ignore(try! owner.take_generated_token())
  ignore(try! owner.take_terminal_notice())
  try! seed_lease.release()

  let (isolated, isolated_lease) = wb_prefix_request(
    912UL,
    [1, 2, 3, 4, 9],
    ReadOnly,
    b"other",
  )
  let isolated_handle = try! owner.admit(isolated, 0UL)
  assert_eq(
    owner.status(isolated_handle.request_id()).unwrap().processed_input_tokens(),
    0,
  )
  ignore(try! owner.cancel(isolated_handle))
  assert_eq((try! owner.take_terminal_notice()).cached_input_tokens(), 0)
  try! isolated_lease.release()

  let before = owner.luna_telemetry().prefix_lookups()
  let (disabled, disabled_lease) = wb_prefix_request(
    913UL,
    [1, 2, 3, 4, 9],
    Disabled,
    b"scope",
  )
  let disabled_handle = try! owner.admit(disabled, 0UL)
  assert_eq(owner.luna_telemetry().prefix_lookups(), before)
  ignore(try! owner.cancel(disabled_handle))
  assert_eq((try! owner.take_terminal_notice()).cached_input_tokens(), 0)
  try! disabled_lease.release()
}

///|
test "waiting evidence pins no pages or radix entries through cancellation" {
  let owner = wb_prefix_scheduler()
  let (seed, seed_lease) = wb_prefix_request(
    914UL,
    [1, 2, 3, 4, 5],
    ReadWrite,
    b"scope",
  )
  ignore(try! owner.admit(seed, 0UL))
  wb_complete_one_token_request(owner, 7)
  ignore(try! owner.take_generated_token())
  ignore(try! owner.take_terminal_notice())
  try! seed_lease.release()

  let (waiting, waiting_lease) = wb_prefix_request(
    915UL,
    [1, 2, 3, 4, 9],
    ReadOnly,
    b"scope",
  )
  let handle = try! owner.admit(waiting, 0UL)
  let slot = handle.slot().index()
  let entry = try! @radix.PrefixEntryId::new(
    index=owner.prefix_candidate_entry_indices[slot],
    generation=owner.prefix_candidate_entry_generations[slot],
  )
  assert_eq(owner.prefix_entry_indices[slot], NO_INDEX)
  assert_eq(owner.prefix_matched_tokens[slot], 0)
  assert_eq(try! owner.prefix_index.unwrap().active_reference_count(entry), 0)
  assert_eq(owner.allocated_block_table_count(), 0)
  assert_eq(owner.evict_one_prefix(), 2)
  assert_eq(owner.luna_telemetry().prefix_entries(), 0)
  assert_eq(owner.luna_telemetry().kv_pages_used(), 0)
  ignore(try! owner.cancel(handle))
  assert_eq(owner.allocated_block_table_count(), 0)
  assert_eq(owner.prefix_candidate_entry_indices[slot], NO_INDEX)
  assert_eq(owner.evict_one_prefix(), 0)
  assert_eq(owner.luna_telemetry().prefix_entries(), 0)
  assert_eq(owner.luna_telemetry().kv_pages_used(), 0)
  ignore(try! owner.take_terminal_notice())
  try! waiting_lease.release()
}

///|
test "stale waiting evidence clears once and falls back uncached" {
  let owner = wb_prefix_scheduler()
  let (seed, seed_lease) = wb_prefix_request(
    916UL,
    [1, 2, 3, 4, 5],
    ReadWrite,
    b"scope",
  )
  ignore(try! owner.admit(seed, 0UL))
  wb_complete_one_token_request(owner, 7)
  ignore(try! owner.take_generated_token())
  ignore(try! owner.take_terminal_notice())
  try! seed_lease.release()

  let (waiting, waiting_lease) = wb_prefix_request(
    917UL,
    [1, 2, 3, 4, 9],
    ReadOnly,
    b"scope",
  )
  let handle = try! owner.admit(waiting, 0UL)
  let slot = handle.slot().index()
  assert_eq(owner.prefix_candidate_tokens[slot], 4)
  assert_eq(owner.luna_telemetry().prefix_hits(), 0UL)
  assert_eq(owner.evict_one_prefix(), 2)
  let plan = wb_next_plan(owner)
  assert_eq(try! plan.prefill_row(0).sequence_token_start(), 0)
  assert_eq(owner.prefix_candidate_entry_indices[slot], NO_INDEX)
  assert_eq(owner.luna_telemetry().prefix_hits(), 0UL)
  assert_eq(owner.luna_telemetry().prefix_misses(), 2UL)
  ignore(try! owner.complete(plan, wb_prefill_completion(owner, plan)))
  ignore(try! owner.cancel(handle))
  ignore(try! owner.take_terminal_notice())
  try! waiting_lease.release()
}

///|
test "waiting expiry clears logical evidence without physical release" {
  let owner = wb_prefix_scheduler()
  let (seed, seed_lease) = wb_prefix_request(
    918UL,
    [1, 2, 3, 4, 5],
    ReadWrite,
    b"scope",
  )
  ignore(try! owner.admit(seed, 0UL))
  wb_complete_one_token_request(owner, 7)
  ignore(try! owner.take_generated_token())
  ignore(try! owner.take_terminal_notice())
  try! seed_lease.release()
  let (waiting, waiting_lease) = wb_prefix_request(
    919UL,
    [1, 2, 3, 4, 9],
    ReadOnly,
    b"scope",
  )
  let handle = try! owner.admit(waiting, 0UL)
  let slot = handle.slot().index()
  assert_eq(owner.prefix_candidate_tokens[slot], 4)
  assert_eq(owner.prefix_entry_indices[slot], NO_INDEX)
  assert_true((try! owner.expire_exact(handle, 100UL)) is Some(_))
  assert_eq(owner.prefix_candidate_entry_indices[slot], NO_INDEX)
  assert_eq(owner.evict_one_prefix(), 2)
  ignore(try! owner.take_terminal_notice())
  try! waiting_lease.release()
}

///|
test "descendant publication retains only newly adopted page anchors" {
  let owner = wb_prefix_scheduler()
  let (seed, seed_lease) = wb_prefix_request(
    931UL,
    [1, 2, 3, 4, 5],
    ReadWrite,
    b"scope",
  )
  ignore(try! owner.admit(seed, 0UL))
  wb_complete_one_token_request(owner, 7)
  ignore(try! owner.take_generated_token())
  ignore(try! owner.take_terminal_notice())
  try! seed_lease.release()

  let (descendant, descendant_lease) = wb_prefix_request(
    932UL,
    [1, 2, 3, 4, 6, 7, 8],
    ReadWrite,
    b"scope",
  )
  ignore(try! owner.admit(descendant, 0UL))
  let intermediate = wb_next_plan(owner)
  assert_eq(try! intermediate.prefill_row(0).tokens().length(), 3)
  ignore(
    try! owner.complete(
      intermediate,
      wb_final_completion(owner, intermediate, 7),
    ),
  )
  ignore(try! owner.take_generated_token())
  ignore(try! owner.take_terminal_notice())
  try! descendant_lease.release()
  let telemetry = owner.luna_telemetry()
  assert_eq(telemetry.prefix_publications(), 2UL)
  assert_eq(telemetry.prefix_entries(), 2)
  assert_eq(telemetry.prefix_pages(), 3)
  assert_eq(telemetry.kv_pages_used(), 3)
}

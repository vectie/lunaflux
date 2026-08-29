///|
fn wb_service_identity() -> @spec.ModelIdentity {
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
fn wb_other_identity() -> @spec.ModelIdentity {
  try! @spec.ModelIdentity::new(
    try! @spec.ContentDigest::from_sha256(
      "1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
    wb_service_identity().plan(),
  )
}

///|
fn wb_service_digest_hex() -> String {
  "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff"
}

///|
fn wb_other_digest_hex() -> String {
  "10112233445566778899aabbccddeeff00112233445566778899aabbccddeeff"
}

///|
fn wb_service_source_digest_hex() -> String {
  "112233445566778899aabbccddeeff00112233445566778899aabbccddeeff00"
}

///|
fn wb_other_source_digest_hex() -> String {
  "212233445566778899aabbccddeeff00112233445566778899aabbccddeeff00"
}

///|
fn wb_service_inference_limits(
  max_text_bytes? : Int = 64,
) -> @inference.InferenceLimits {
  try! @inference.InferenceLimits::new(
    max_text_bytes~,
    max_input_tokens=4,
    max_new_tokens=2,
    max_context_tokens=4,
    max_token_id=31,
    max_stop_token_ids=1,
    max_stop_strings=1,
    max_stop_string_bytes=1,
    max_trace_bytes=8,
    max_cache_scope_bytes=8,
    max_decoded_delta_bytes=8,
    max_deadline_millis=1000UL,
    max_top_k=32,
    max_temperature=4.0,
  )
}

///|
fn wb_service_worker_limits(
  max_plan_tokens? : Int = 1,
) -> @worker_protocol.WorkerProtocolLimits {
  try! @worker_protocol.WorkerProtocolLimits::new(
    max_prefill_rows=1,
    max_decode_rows=1,
    max_plan_rows=1,
    max_plan_tokens~,
    max_plan_pages=2,
    max_capabilities=2,
    max_completion_slots=1,
    max_sequence_tokens=4,
    max_token_id=31,
  )
}

///|
fn wb_service_scheduler() -> @core.Scheduler {
  let worker = wb_service_worker_limits()
  let scheduling = try! @scheduler_config.new(
    step_token_budget=1,
    max_active_requests=2,
    max_waiting_requests=4,
    prefill_chunk_tokens=1,
    emergency_decode_page_reserve=1,
    waiting_age_threshold_steps=2,
    output_event_capacity=1,
  )
  let cache = try! @cache.new(
    tokens_per_page=2,
    total_page_count=8,
    block_table_pages_per_request=2,
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
    max_sequence_tokens=4,
    vocabulary_size=32,
    hidden_size=8,
    intermediate_size=16,
    attention_head_count=2,
    kv_head_count=1,
    head_dimension=4,
  )
  let capacity = try! @runtime_resolved.resolve(
    scheduling, cache, shapes, worker,
  )
  let generation = @worker_protocol.ModelPlanGeneration::first()
  let recipe = try! @worker_protocol.RowCapabilityRecipe::new(
    identity=wb_service_identity(),
    model_plan_generation=generation,
    [@plan.KernelCapabilityId::embedding_lookup()],
    worker,
  )
  try! @core.Scheduler::new(
    capacity,
    wb_service_identity(),
    generation,
    recipe,
    recipe,
    recipe,
  )
}

///|
fn wb_service_binding() -> WorkerServiceBinding {
  let digest = try! @worker_wire.WorkerBootstrapDigest::from_sha256(
    wb_service_digest_hex(),
  )
  let source_digest = try! @worker_wire.WorkerBootstrapSourceDigest::from_sha256(
    wb_service_source_digest_hex(),
  )
  try! WorkerServiceBinding::new(
    bootstrap_digest=digest,
    bootstrap_source_digest=source_digest,
    device_ordinal=0,
    inference_limits=wb_service_inference_limits(),
  )
}

///|
fn wb_worker_service_identity() -> WorkerServiceIdentity {
  let scheduler = wb_service_scheduler()
  let binding = wb_service_binding()
  try! worker_service_identity(
    scheduler.model_identity(),
    scheduler.model_plan_generation().value(),
    binding.bootstrap_source_digest(),
    scheduler.worker_protocol_limits(),
    binding.inference_limits(),
  )
}

///|
fn wb_startup_contract(
  identity? : @spec.ModelIdentity = wb_service_identity(),
  bootstrap_digest_hex? : String = wb_service_digest_hex(),
  bootstrap_source_digest_hex? : String = wb_service_source_digest_hex(),
  device_ordinal? : Int = 0,
  model_generation_value? : UInt64 = 1UL,
  worker_limits? : @worker_protocol.WorkerProtocolLimits = wb_service_worker_limits(),
  inference_limits? : @inference.InferenceLimits = wb_service_inference_limits(),
  has_predecessor? : Bool = false,
  predecessor_value? : UInt64 = 0UL,
) -> @worker_wire.WorkerStartupContract {
  let digest = try! @worker_wire.WorkerBootstrapDigest::from_sha256(
    bootstrap_digest_hex,
  )
  let source_digest = try! @worker_wire.WorkerBootstrapSourceDigest::from_sha256(
    bootstrap_source_digest_hex,
  )
  try! @worker_wire.WorkerStartupContract::new(
    identity~,
    bootstrap_digest=digest,
    bootstrap_source_digest=source_digest,
    device_ordinal~,
    model_generation_value~,
    worker_limits~,
    inference_limits~,
    has_predecessor~,
    predecessor_value~,
  )
}

///|
fn expect_binding_rule(
  scheduler : @core.Scheduler,
  binding : WorkerServiceBinding,
  startup : @worker_wire.WorkerStartupContract,
  expected : WorkerServiceRule,
) -> Unit raise {
  try
    validate_startup_contract(
      scheduler,
      wb_worker_service_identity(),
      binding,
      startup,
    )
  catch {
    Invalid(rule) => assert_eq(rule, expected)
    _ => fail("unexpected worker-service binding error")
  } noraise {
    _ => fail("mismatched startup joined the worker service")
  }
}

///|
test "exact startup contract matches independent service binding" {
  validate_startup_contract(
    wb_service_scheduler(),
    wb_worker_service_identity(),
    wb_service_binding(),
    wb_startup_contract(),
  )
}

///|
test "startup mismatch rules are exact and validation does not mutate scheduler" {
  let scheduler = wb_service_scheduler()
  let binding = wb_service_binding()
  let free_pages = scheduler.free_page_count()
  let live_requests = scheduler.live_request_count()
  let submitted = scheduler.last_submitted_plan_sequence()
  let retired = scheduler.last_retired_plan_sequence()
  expect_binding_rule(
    scheduler,
    binding,
    wb_startup_contract(identity=wb_other_identity()),
    Identity,
  )
  expect_binding_rule(
    scheduler,
    binding,
    wb_startup_contract(bootstrap_digest_hex=wb_other_digest_hex()),
    BootstrapIdentity,
  )
  expect_binding_rule(
    scheduler,
    binding,
    wb_startup_contract(
      bootstrap_source_digest_hex=wb_other_source_digest_hex(),
    ),
    BootstrapSourceIdentity,
  )
  expect_binding_rule(
    scheduler,
    binding,
    wb_startup_contract(device_ordinal=1),
    DeviceAssignment,
  )
  expect_binding_rule(
    scheduler,
    binding,
    wb_startup_contract(model_generation_value=2UL),
    Generation,
  )
  expect_binding_rule(
    scheduler,
    binding,
    wb_startup_contract(
      worker_limits=wb_service_worker_limits(max_plan_tokens=2),
    ),
    WorkerEnvelope,
  )
  expect_binding_rule(
    scheduler,
    binding,
    wb_startup_contract(
      inference_limits=wb_service_inference_limits(max_text_bytes=65),
    ),
    InferenceEnvelope,
  )
  expect_binding_rule(
    scheduler,
    binding,
    wb_startup_contract(has_predecessor=true, predecessor_value=1UL),
    Sequence,
  )
  assert_eq(scheduler.free_page_count(), free_pages)
  assert_eq(scheduler.live_request_count(), live_requests)
  assert_eq(scheduler.last_submitted_plan_sequence(), submitted)
  assert_eq(scheduler.last_retired_plan_sequence(), retired)
}

///|
test "service binding rejects invalid visible device ordinal distinctly" {
  let digest = try! @worker_wire.WorkerBootstrapDigest::from_sha256(
    wb_service_digest_hex(),
  )
  let source_digest = try! @worker_wire.WorkerBootstrapSourceDigest::from_sha256(
    wb_service_source_digest_hex(),
  )
  try
    WorkerServiceBinding::new(
      bootstrap_digest=digest,
      bootstrap_source_digest=source_digest,
      device_ordinal=-1,
      inference_limits=wb_service_inference_limits(),
    )
  catch {
    Invalid(DeviceAssignment) => ()
    _ => fail("unexpected invalid service binding error")
  } noraise {
    _ => fail("negative visible device ordinal was accepted")
  }
}

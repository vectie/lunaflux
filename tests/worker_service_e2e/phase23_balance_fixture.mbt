///|
const PHASE23_BALANCE_REQUEST_COUNT : Int = 10000

///|
const PHASE23_BALANCE_WAVE_SIZE : Int = 10

///|
const PHASE23_BALANCE_PAGE_COUNT : Int = 16

///|
fn phase23_balance_worker_limits() -> @worker_protocol.WorkerProtocolLimits raise {
  @worker_protocol.WorkerProtocolLimits::new(
    max_prefill_rows=4,
    max_decode_rows=4,
    max_plan_rows=4,
    max_plan_tokens=4,
    max_plan_pages=8,
    max_capabilities=4,
    max_completion_slots=4,
    max_sequence_tokens=4,
    max_token_id=31,
  )
}

///|
fn phase23_balance_bootstrap_source(
  fixture_root : StringView,
) -> @worker_wire.EncodedBootstrapSource raise {
  @worker_wire.admit_bootstrap_source(
    @worker_wire.WorkerBootstrapRoots::new(
      model_root=fixture_root,
      kernel_root=fixture_root,
    ),
    @worker_wire.WorkerBootstrapModelSource::new(
      config_locator="config.json",
      weights_locator="weights.bin",
      config_digest_sha256="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
    @worker_wire.WorkerBootstrapKernelSource::new(
      manifest_locator="manifest.json",
      manifest_digest_sha256="abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
    ),
    @worker_wire.WorkerBootstrapExecution::new(
      compute_major=8,
      compute_minor=0,
      supports_bf16=true,
      supports_cublas_lt=true,
      tokens_per_page=16,
      total_page_count=PHASE23_BALANCE_PAGE_COUNT,
    ),
    @worker_wire.WorkerBootstrapCeilings::new(
      max_model_config_bytes=1024,
      max_weight_file_bytes=1024L,
      max_weight_arena_bytes=1024L,
      max_activation_arena_bytes=1024L,
      max_kv_arena_bytes=1024L,
      max_execution_manifest_bytes=1024L,
      max_module_bytes=1024L,
      max_total_module_bytes=1024L,
    ),
  )
}

///|
fn phase23_balance_blueprint() -> @scheduler.SchedulerBlueprint raise {
  let scheduling = @scheduler_config.new(
    step_token_budget=4,
    max_active_requests=4,
    max_waiting_requests=8,
    prefill_chunk_tokens=1,
    emergency_decode_page_reserve=1,
    waiting_age_threshold_steps=2,
    output_event_capacity=4,
  )
  let cache = @cache.new(
    tokens_per_page=2,
    total_page_count=PHASE23_BALANCE_PAGE_COUNT,
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
  let shapes = @plan.ShapeConstraints::new(
    max_batch_rows=4,
    max_sequence_tokens=4,
    vocabulary_size=32,
    hidden_size=8,
    intermediate_size=16,
    attention_head_count=2,
    kv_head_count=1,
    head_dimension=4,
  )
  let limits = phase23_balance_worker_limits()
  let capacity = @runtime_resolved.resolve(scheduling, cache, shapes, limits)
  let generation = @worker_protocol.ModelPlanGeneration::first()
  let recipe = @worker_protocol.RowCapabilityRecipe::new(
    identity=identity(),
    model_plan_generation=generation,
    [@plan.KernelCapabilityId::embedding_lookup()],
    limits,
  )
  let blueprint = @scheduler.SchedulerBlueprint::new(
    capacity,
    identity(),
    generation,
    recipe,
    recipe,
    recipe,
  )
  assert_eq(blueprint.request_slot_capacity(), 12)
  blueprint
}

///|
fn phase23_balance_startup(
  fixture_root : StringView,
) -> @worker_wire.WorkerStartupContract raise {
  @worker_wire.WorkerStartupContract::new(
    identity=identity(),
    bootstrap_digest=@worker_wire.WorkerBootstrapDigest::from_sha256(
      "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff",
    ),
    bootstrap_source_digest=phase23_balance_bootstrap_source(fixture_root).digest(),
    device_ordinal=0,
    model_generation_value=1UL,
    worker_limits=phase23_balance_worker_limits(),
    inference_limits=inference_limits(),
    has_predecessor=false,
    predecessor_value=0UL,
  )
}

///|
fn phase23_balance_binding(
  fixture_root : StringView,
) -> @worker_service.WorkerServiceBinding raise {
  @worker_service.WorkerServiceBinding::new(
    bootstrap_digest=@worker_wire.WorkerBootstrapDigest::from_sha256(
      "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff",
    ),
    bootstrap_source_digest=phase23_balance_bootstrap_source(fixture_root).digest(),
    device_ordinal=0,
    inference_limits=inference_limits(),
  )
}

///|
fn phase23_balance_process_limits() -> @worker_process.WorkerProcessLimits raise {
  @worker_process.WorkerProcessLimits::new(
    max_frame_bytes=@worker_wire.bootstrap_source_capacity(),
    startup_io_timeout_millis=5000,
    io_timeout_millis=5000,
    shutdown_timeout_millis=100,
  )
}

///|
fn phase23_balance_service(
  executable : Bytes,
  fixture_root : StringView,
) -> @worker_service.WorkerService raise {
  let model = @approved_fs.ApprovedRoot::open_absolute(fixture_root)
  let kernel = @approved_fs.ApprovedRoot::open_absolute(fixture_root)
  let prepared = @worker_service.prepare_owned_approved(
    phase23_balance_blueprint(),
    phase23_balance_binding(fixture_root),
    @worker_executable_fixture.admit(executable),
    phase23_balance_startup(fixture_root),
    phase23_balance_bootstrap_source(fixture_root),
    phase23_balance_process_limits(),
    model,
    kernel,
    restart_policy(),
  )
  model.close()
  kernel.close()
  assert_eq(prepared.state(), OwnedServiceReady)
  prepared.take_raw_ready()
}

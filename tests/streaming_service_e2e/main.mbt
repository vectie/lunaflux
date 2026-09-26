///|
fn run(executable : Bytes) -> Unit raise {
  let digest = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  let identity = @spec.ModelIdentity::new(
    @spec.ContentDigest::from_sha256(digest),
    @spec.PlanDigest::from_sha256(digest),
  )
  let inference = @inference.InferenceLimits::new(
    max_text_bytes=64,
    max_input_tokens=4,
    max_new_tokens=1,
    max_context_tokens=4,
    max_token_id=31,
    max_stop_token_ids=1,
    max_stop_strings=1,
    max_stop_string_bytes=1,
    max_trace_bytes=8,
    max_cache_scope_bytes=8,
    max_decoded_delta_bytes=8,
    max_deadline_millis=60000UL,
    max_top_k=32,
    max_temperature=4.0,
  )
  let worker = @worker_protocol.WorkerProtocolLimits::new(
    max_prefill_rows=1,
    max_decode_rows=1,
    max_plan_rows=1,
    max_plan_tokens=1,
    max_plan_pages=2,
    max_capabilities=2,
    max_completion_slots=1,
    max_sequence_tokens=4,
    max_token_id=31,
  )
  let scheduling = @scheduler_config.new(
    step_token_budget=1,
    max_active_requests=1,
    max_waiting_requests=1,
    prefill_chunk_tokens=1,
    emergency_decode_page_reserve=1,
    waiting_age_threshold_steps=2,
    output_event_capacity=2,
  )
  let cache = @cache.new(
    tokens_per_page=2,
    total_page_count=8,
    block_table_pages_per_request=2,
    prefix_enabled=true,
    max_prefix_entries=2,
    max_prefix_nodes=16,
    max_prefix_tokens_per_entry=4,
    max_prefix_pages=4,
    max_prefix_scope_bytes=8,
    max_active_references_per_page=4,
    max_cached_references_per_page=1,
    layout_version=1,
  )
  let shapes = @plan.ShapeConstraints::new(
    max_batch_rows=1,
    max_sequence_tokens=4,
    vocabulary_size=32,
    hidden_size=8,
    intermediate_size=16,
    attention_head_count=2,
    kv_head_count=1,
    head_dimension=4,
  )
  let recipe = @worker_protocol.RowCapabilityRecipe::new(
    identity~,
    model_plan_generation=@worker_protocol.ModelPlanGeneration::first(),
    [@plan.KernelCapabilityId::embedding_lookup()],
    worker,
  )
  let scheduler = @scheduler.Scheduler::new(
    @runtime_resolved.resolve(scheduling, cache, shapes, worker),
    identity,
    @worker_protocol.ModelPlanGeneration::first(),
    recipe,
    recipe,
    recipe,
  )
  let source = @worker_wire.admit_bootstrap_source(
    @worker_wire.WorkerBootstrapRoots::new(
      model_root="/private/tmp",
      kernel_root="/private/tmp",
    ),
    @worker_wire.WorkerBootstrapModelSource::new(
      config_locator="config.json",
      weights_locator="weights.bin",
      config_digest_sha256=digest,
    ),
    @worker_wire.WorkerBootstrapKernelSource::new(
      manifest_locator="manifest.json",
      manifest_digest_sha256=digest,
    ),
    @worker_wire.WorkerBootstrapExecution::new(
      compute_major=8,
      compute_minor=0,
      supports_bf16=true,
      supports_cublas_lt=true,
      tokens_per_page=2,
      total_page_count=8,
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
  let bootstrap = @worker_wire.WorkerBootstrapDigest::from_sha256(digest)
  let startup = @worker_wire.WorkerStartupContract::new(
    identity~,
    bootstrap_digest=bootstrap,
    bootstrap_source_digest=source.digest(),
    device_ordinal=0,
    model_generation_value=1UL,
    worker_limits=worker,
    inference_limits=inference,
    has_predecessor=false,
    predecessor_value=0UL,
    streaming=@streaming_ir.Capability::new(
      slots=2,
      host_bytes=128L,
      page_bytes=64L,
      page_count=8,
      reserve_pages=8,
      restore_ns=100L,
      recompute_ns=1000L,
      separate_host_memory=true,
    ),
  )
  let root = @approved_fs.ApprovedRoot::open_absolute("/private/tmp")
  let prepared = @worker_process.prepare_with_approved_executable(
    @worker_executable_fixture.admit(executable),
    startup,
    source,
    @worker_process.WorkerProcessLimits::new(
      max_frame_bytes=@worker_wire.bootstrap_source_capacity(),
      startup_io_timeout_millis=5000,
      io_timeout_millis=5000,
      shutdown_timeout_millis=5000,
    ),
    root,
    root,
  )
  root.close()
  guard prepared is ReadyRootBoundProcess(process) else {
    raise Failure("streaming fixture preparation failed")
  }
  let service = @worker_service.new_fixture(
    scheduler,
    @worker_service.WorkerServiceBinding::new(
      bootstrap_digest=bootstrap,
      bootstrap_source_digest=source.digest(),
      device_ordinal=0,
      inference_limits=inference,
    ),
    process,
  )
  let storage = @inference.LunaRequestSemanticStorage::new(
    inference,
    @inference.LunaRequestSemanticStepBudget::new(17),
  )
  let write = storage.begin(
    stop_token_count=0,
    stop_string_count=0,
    cache_scope_byte_count=1,
    cache_permission=ReadWrite,
  )
  write.push_cache_scope_byte(b'x')
  let work = write.finish()
  while work.progress().is_pending() {

  }
  let lease = work.take_lease()
  let clock_output = Ref(0UL)
  assert_true(@monotonic_clock.MonotonicClock::system().read_into(clock_output))
  let now = clock_output.val
  let request = @scheduler.TokenizedRequest::new_with_prefix(
    request_id=@inference.RequestId::new(1UL),
    model_identity=identity,
    input_tokens=@inference.TokenBuffer::new([1, 2, 3], inference),
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=@inference.SamplingSeed::new(1UL),
    stop_tokens=lease.stop_token_view(),
    stream_preference=Streaming,
    effective_limits=@inference.EffectiveLimits::new(
      max_new_tokens=1,
      context_ceiling=4,
      inference,
    ),
    admission_deadline=@inference.AdmissionDeadline::from_receipt(
      now,
      @inference.DeadlineBudget::from_milliseconds(60000UL, inference),
    ),
    tokenizer_digest=@tokenizer.TokenizerDigest::parse(digest),
    prefix_semantic=lease.prefix_view(),
  )
  ignore(service.admit(request, now))
  drive_request(service)
  assert_eq(service.take_generated_token().token_id(), 0)
  assert_eq(service.take_terminal_notice().cached_input_tokens(), 0)
  drive_idle(service)
  assert_eq(service.free_page_count(), 8)
  ignore(service.admit(request, now))
  drive_request(service)
  assert_eq(service.take_generated_token().token_id(), 0)
  assert_eq(service.take_terminal_notice().cached_input_tokens(), 2)
  drive_idle(service)
  let handle = service.admit(request, now)
  ignore(service.progress())
  assert_true(service.outstanding_count() > 0)
  ignore(service.cancel(handle))
  drive_idle(service)
  assert_eq(service.take_terminal_notice().cached_input_tokens(), 0)
  assert_eq(service.free_page_count(), 8)
  service.shutdown_clean()
  lease.release()
  println(
    "Streaming service: cold reuse, output equality, cancellation and clean shutdown passed",
  )
}

///|
fn drive_request(service : @worker_service.WorkerService) -> Unit raise {
  for _ in 0..<1000000 {
    ignore(service.progress())
    if service.live_request_count() == 0 && service.outstanding_count() == 0 {
      return
    }
  }
  raise Failure("request progress exhausted")
}

///|
fn drive_idle(service : @worker_service.WorkerService) -> Unit raise {
  for _ in 0..<1000000 {
    if service.progress() is ServiceIdle && service.outstanding_count() == 0 {
      return
    }
  }
  raise Failure("residency progress exhausted")
}

///|
fn main raise {
  guard @env.args() is [_, executable] else {
    raise Failure("worker executable path required")
  }
  run(@utf8.encode(executable))
}

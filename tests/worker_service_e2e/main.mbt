///|
fn identity() -> @spec.ModelIdentity raise {
  @spec.ModelIdentity::new(
    @spec.ContentDigest::from_sha256(
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
    @spec.PlanDigest::from_sha256(
      "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
    ),
  )
}

///|
fn inference_limits() -> @inference.InferenceLimits raise {
  @inference.InferenceLimits::new(
    max_text_bytes=64,
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
fn worker_limits() -> @worker_protocol.WorkerProtocolLimits raise {
  @worker_protocol.WorkerProtocolLimits::new(
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
}

///|
fn bootstrap_source() -> @worker_wire.EncodedBootstrapSource raise {
  @worker_wire.admit_bootstrap_source(
    @worker_wire.WorkerBootstrapRoots::new(
      model_root="/models",
      kernel_root="/kernels",
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
}

///|
fn capacity() -> @runtime_resolved.ResolvedRuntimeCapacity raise {
  let scheduling = @scheduler_config.new(
    step_token_budget=1,
    max_active_requests=2,
    max_waiting_requests=4,
    prefill_chunk_tokens=1,
    emergency_decode_page_reserve=1,
    waiting_age_threshold_steps=2,
    output_event_capacity=1,
  )
  let cache = @cache.new(
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
  @runtime_resolved.resolve(scheduling, cache, shapes, worker_limits())
}

///|
fn scheduler_blueprint() -> @scheduler.SchedulerBlueprint raise {
  let generation = @worker_protocol.ModelPlanGeneration::first()
  let recipe = @worker_protocol.RowCapabilityRecipe::new(
    identity=identity(),
    model_plan_generation=generation,
    [@plan.KernelCapabilityId::embedding_lookup()],
    worker_limits(),
  )
  @scheduler.SchedulerBlueprint::new(
    capacity(),
    identity(),
    generation,
    recipe,
    recipe,
    recipe,
  )
}

///|
fn scheduler_owner() -> @scheduler.Scheduler raise {
  scheduler_blueprint().instantiate()
}

///|
fn request(
  id : UInt64,
  max_new_tokens? : Int = 1,
) -> @scheduler.TokenizedRequest raise {
  let limits = inference_limits()
  @scheduler.TokenizedRequest::new(
    request_id=@inference.RequestId::new(id),
    model_identity=identity(),
    input_tokens=@inference.TokenBuffer::new([1], limits),
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=@inference.SamplingSeed::new(id),
    stops=@inference.StopConditions::new([], [], limits),
    cache=@inference.CachePolicy::new(
      @inference.CacheScope::from_ascii(b"service", limits),
      Disabled,
    ),
    stream_preference=Streaming,
    effective_limits=@inference.EffectiveLimits::new(
      max_new_tokens~,
      context_ceiling=4,
      limits,
    ),
    admission_deadline=@inference.AdmissionDeadline::from_receipt(
      0UL,
      @inference.DeadlineBudget::from_milliseconds(100UL, limits),
    ),
  )
}

///|
fn online_request(
  id : UInt64,
  receipt : UInt64,
  max_new_tokens? : Int = 1,
  deadline_budget? : UInt64 = 1000UL,
) -> @scheduler.TokenizedRequest raise {
  let limits = inference_limits()
  @scheduler.TokenizedRequest::new(
    request_id=@inference.RequestId::new(id),
    model_identity=identity(),
    input_tokens=@inference.TokenBuffer::new([1], limits),
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=@inference.SamplingSeed::new(id),
    stops=@inference.StopConditions::new([], [], limits),
    cache=@inference.CachePolicy::new(
      @inference.CacheScope::from_ascii(b"online", limits),
      Disabled,
    ),
    stream_preference=Streaming,
    effective_limits=@inference.EffectiveLimits::new(
      max_new_tokens~,
      context_ceiling=4,
      limits,
    ),
    admission_deadline=@inference.AdmissionDeadline::from_receipt(
      receipt,
      @inference.DeadlineBudget::from_milliseconds(deadline_budget, limits),
    ),
  )
}

///|
fn startup() -> @worker_wire.WorkerStartupContract raise {
  @worker_wire.WorkerStartupContract::new(
    identity=identity(),
    bootstrap_digest=@worker_wire.WorkerBootstrapDigest::from_sha256(
      "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff",
    ),
    bootstrap_source_digest=bootstrap_source().digest(),
    device_ordinal=0,
    model_generation_value=1UL,
    worker_limits=worker_limits(),
    inference_limits=inference_limits(),
    has_predecessor=false,
    predecessor_value=0UL,
  )
}

///|
fn service_binding() -> @worker_service.WorkerServiceBinding raise {
  @worker_service.WorkerServiceBinding::new(
    bootstrap_digest=@worker_wire.WorkerBootstrapDigest::from_sha256(
      "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff",
    ),
    bootstrap_source_digest=bootstrap_source().digest(),
    device_ordinal=0,
    inference_limits=inference_limits(),
  )
}

///|
fn foreign_startup() -> @worker_wire.WorkerStartupContract raise {
  @worker_wire.WorkerStartupContract::new(
    identity=@spec.ModelIdentity::new(
      @spec.ContentDigest::from_sha256(
        "1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      ),
      identity().plan(),
    ),
    bootstrap_digest=@worker_wire.WorkerBootstrapDigest::from_sha256(
      "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff",
    ),
    bootstrap_source_digest=bootstrap_source().digest(),
    device_ordinal=0,
    model_generation_value=1UL,
    worker_limits=worker_limits(),
    inference_limits=inference_limits(),
    has_predecessor=false,
    predecessor_value=0UL,
  )
}

///|
fn process_limits() -> @worker_process.WorkerProcessLimits raise {
  @worker_process.WorkerProcessLimits::new(
    max_frame_bytes=@worker_wire.bootstrap_source_capacity(),
    startup_io_timeout_millis=5000,
    io_timeout_millis=5000,
    shutdown_timeout_millis=5000,
  )
}

///|
fn process_owner(
  executable : Bytes,
) -> @worker_process.RootBoundWorkerProcessSupervisor raise {
  let model = @approved_fs.ApprovedRoot::open_absolute("/private/tmp")
  let kernel = @approved_fs.ApprovedRoot::open_absolute("/private/tmp")
  let result = @worker_process.prepare_with_approved_roots(
    executable,
    startup(),
    bootstrap_source(),
    process_limits(),
    model,
    kernel,
  )
  model.close()
  kernel.close()
  match result {
    ReadyRootBoundProcess(owner) => owner
    RootBoundCleanupRequired(_) =>
      raise Failure("worker startup cleanup required")
  }
}

///|
fn progress_to_ready(
  service : @worker_service.WorkerService,
  expected : UInt64,
) -> Unit raise {
  match service.progress() {
    ServiceStarted(sequence) => assert_eq(sequence, expected)
    _ => raise Failure("service did not start the expected exchange")
  }
  let mut attempts = 0
  while attempts < 1000000 {
    match service.progress() {
      ServicePending(sequence) => {
        assert_eq(sequence, expected)
        attempts += 1
      }
      ServiceCompletionReady(sequence) => {
        assert_eq(sequence, expected)
        return
      }
      _ => raise Failure("unexpected service exchange progress")
    }
  }
  raise Failure("service exchange did not complete")
}

///|
fn commit_ready(
  service : @worker_service.WorkerService,
  expected : UInt64,
) -> Unit raise {
  match service.progress() {
    ServiceCommitted(sequence) => assert_eq(sequence, expected)
    _ => raise Failure("service did not commit the expected exchange")
  }
}

///|
fn complete_root_exchange(
  owner : @worker_process.RootBoundWorkerProcessSupervisor,
) -> Unit raise {
  let mut attempts = 0
  while attempts < 1000000 {
    match owner.progress_exchange() {
      ExchangePending => attempts += 1
      ExchangeCompleted => return
    }
  }
  raise Failure("direct root exchange did not complete")
}

///|
fn run(executable : Bytes) -> Unit raise {
  run_owned_startup_failure_evidence()
  let scheduler = scheduler_owner()
  let foreign_model = @approved_fs.ApprovedRoot::open_absolute("/private/tmp")
  let foreign_kernel = @approved_fs.ApprovedRoot::open_absolute("/private/tmp")
  let foreign_result = @worker_process.prepare_with_approved_roots(
    executable,
    foreign_startup(),
    bootstrap_source(),
    process_limits(),
    foreign_model,
    foreign_kernel,
  )
  foreign_model.close()
  foreign_kernel.close()
  let foreign = match foreign_result {
    ReadyRootBoundProcess(owner) => owner
    RootBoundCleanupRequired(_) =>
      raise Failure("foreign worker cleanup required")
  }
  try @worker_service.new_fixture(scheduler, service_binding(), foreign) catch {
    @worker_service.Invalid(Identity) => ()
    _ => raise Failure("unexpected foreign worker identity error")
  } noraise {
    _ => raise Failure("foreign worker identity joined the scheduler")
  }
  foreign.close()
  let owned_model = @approved_fs.ApprovedRoot::open_absolute("/private/tmp")
  let owned_kernel = @approved_fs.ApprovedRoot::open_absolute("/private/tmp")
  let owned = @worker_service.prepare_owned(
    scheduler_blueprint(),
    service_binding(),
    executable,
    startup(),
    bootstrap_source(),
    process_limits(),
    owned_model,
    owned_kernel,
  )
  owned_model.close()
  owned_kernel.close()
  assert_eq(owned.state(), OwnedServiceReady)
  let service = owned.take_raw_ready()
  try owned.take_raw_ready() catch {
    OwnedPreparationFailure(OwnedServiceFailure(Invalid(Lifecycle))) => ()
    _ => raise Failure("unexpected repeated owned-service extraction error")
  } noraise {
    _ => raise Failure("owned-service preparation extracted twice")
  }
  ignore(service.admit(request(101UL), 0UL))
  ignore(service.admit(request(102UL), 0UL))
  progress_to_ready(service, 1UL)
  commit_ready(service, 1UL)
  progress_to_ready(service, 2UL)
  try commit_ready(service, 2UL) catch {
    @worker_service.Scheduler(
      Invalid(field=OutputCapacity, issue=Backpressured)
    ) => ()
    _ => raise Failure("unexpected service publication error")
  } noraise {
    _ => raise Failure("full output ring accepted second completion")
  }
  assert_eq(service.outstanding_count(), 1)
  let first_publication = service.take_next_publication()
  assert_eq(first_publication.kind(), GeneratedTokenPublication)
  assert_eq(
    first_publication.generated_token_notice().request_id(),
    @inference.RequestId::new(101UL),
  )
  let first_terminal = service.take_next_publication()
  assert_eq(first_terminal.kind(), TerminalPublication)
  assert_eq(first_terminal.terminal_notice().reason(), MaximumOutputReached)
  commit_ready(service, 2UL)
  let second_publication = service.take_next_publication()
  assert_eq(second_publication.kind(), GeneratedTokenPublication)
  assert_eq(
    second_publication.generated_token_notice().request_id(),
    @inference.RequestId::new(102UL),
  )
  assert_eq(
    service.take_next_publication().terminal_notice().reason(),
    MaximumOutputReached,
  )

  // Keep one decoded request live across worker loss. The replacement owns a
  // fresh KV arena, so restart must terminally invalidate the survivor before
  // it can become ready even with no outstanding transport frame.
  ignore(service.admit(request(103UL, max_new_tokens=2), 0UL))
  progress_to_ready(service, 3UL)
  commit_ready(service, 3UL)
  ignore(service.take_generated_token())
  assert_eq(service.live_request_count(), 1)
  assert_eq(service.allocated_block_table_count(), 1)
  try service.shutdown_clean() catch {
    Invalid(Lifecycle) => ()
    _ => raise Failure("unexpected live-request shutdown rejection")
  } noraise {
    _ => raise Failure("clean shutdown discarded a live request")
  }
  service.begin_recovery()
  assert_eq(service.lifecycle(), ServiceDeviceInvalidationRequired)
  try service.restart() catch {
    Invalid(Lifecycle) => ()
    _ => raise Failure("unexpected pre-invalidation restart rejection")
  } noraise {
    _ => raise Failure("replacement started before device invalidation")
  }
  assert_eq(service.invalidate_device_state(), 1)
  assert_eq(service.lifecycle(), ServiceRestartReady)
  match service.restart() {
    Restarted => ()
    RestartCleanupRequired =>
      raise Failure("replacement worker cleanup required")
  }
  assert_eq(service.take_terminal_notice().reason(), WorkerFailed)
  assert_eq(service.live_request_count(), 0)
  assert_eq(service.free_page_count(), 8)
  assert_eq(service.allocated_block_table_count(), 0)
  assert_eq(service.binding(), service_binding())
  ignore(service.admit(request(104UL), 0UL))
  progress_to_ready(service, 4UL)
  commit_ready(service, 4UL)
  assert_eq(
    service.take_generated_token().request_id(),
    @inference.RequestId::new(104UL),
  )
  assert_eq(service.take_terminal_notice().reason(), MaximumOutputReached)
  assert_eq(service.outstanding_count(), 0)
  assert_eq(service.live_request_count(), 0)
  assert_eq(service.free_page_count(), 8)
  assert_eq(service.allocated_block_table_count(), 0)
  service.shutdown_clean()
  assert_true(service.lifecycle() == ServiceClosed)

  run_terminal_drain_evidence(executable)
  run_reservation_classification_evidence(executable)
  run_online_lease_evidence(executable)
}

///|
fn main {
  match @env.args() {
    [_, executable] => try! run(@utf8.encode(executable))
    _ => println("worker executable path is required")
  }
}

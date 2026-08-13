///|
fn worker_limits() -> @worker_protocol.WorkerProtocolLimits raise {
  @worker_protocol.WorkerProtocolLimits::new(
    max_prefill_rows=4,
    max_decode_rows=4,
    max_plan_rows=8,
    max_plan_tokens=32,
    max_plan_pages=32,
    max_capabilities=32,
    max_completion_slots=8,
    max_sequence_tokens=64,
    max_token_id=1024,
  )
}

///|
fn inference_limits() -> @inference.InferenceLimits raise {
  @inference.InferenceLimits::new(
    max_text_bytes=1024,
    max_input_tokens=64,
    max_new_tokens=64,
    max_context_tokens=64,
    max_token_id=1024,
    max_stop_token_ids=8,
    max_stop_strings=1,
    max_stop_string_bytes=1,
    max_trace_bytes=64,
    max_cache_scope_bytes=64,
    max_decoded_delta_bytes=64,
    max_deadline_millis=1000UL,
    max_top_k=32,
    max_temperature=4.0,
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
fn startup_contract() -> @worker_wire.WorkerStartupContract raise {
  @worker_wire.WorkerStartupContract::new(
    identity=@spec.ModelIdentity::new(
      @spec.ContentDigest::from_sha256(
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      ),
      @spec.PlanDigest::from_sha256(
        "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
      ),
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
fn page_allocator() -> @page_allocator.PageAllocator raise {
  @page_allocator.PageAllocator::new(
    @page_allocator.PageAllocatorLimits::new(
      page_capacity=8,
      max_run_pages=8,
      max_active_references_per_page=8,
      max_cached_references_per_page=8,
    ),
  )
}

///|
fn plan(
  sequence : UInt64,
  request : UInt64,
) -> @worker_protocol.SubmittedSchedulePlan raise {
  let limits = worker_limits()
  let owner = @worker_protocol.SchedulePlanBuffer::new(limits)
  owner.append_prefill(
    request_id=@inference.RequestId::new(request),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_start=0,
    tokens=[1, 2],
    pages=[page_allocator().allocate_one()],
    capabilities=[@plan.KernelCapabilityId::embedding_lookup()],
    completion_slot=@worker_protocol.CompletionSlot::new(0, limits),
  )
  owner.submit_wire(
    sequence_value=sequence,
    model_plan_generation_value=1UL,
    token_budget=2,
    has_previous=sequence > 1UL,
    previous_sequence_value=if sequence > 1UL { sequence - 1UL } else { 0UL },
    loaded_generation_value=1UL,
  )
}

///|
fn start_worker(
  executable : Bytes,
  startup : @worker_wire.WorkerStartupContract,
) -> @worker_process.WorkerProcessSupervisor raise {
  match
    @worker_process.prepare(
      executable,
      startup,
      bootstrap_source(),
      @worker_process.WorkerProcessLimits::new(
        max_frame_bytes=@worker_wire.bootstrap_source_capacity(),
        startup_io_timeout_millis=5000,
        io_timeout_millis=5000,
        shutdown_timeout_millis=5000,
      ),
    ) {
    ReadyProcess(owner) => owner
    CleanupRequired(_) => raise Failure("worker startup cleanup required")
  }
}

///|
fn start_root_worker(
  executable : Bytes,
) -> @worker_process.RootBoundWorkerProcessSupervisor raise {
  let model = @approved_fs.ApprovedRoot::open_absolute("/private/tmp")
  let kernel = @approved_fs.ApprovedRoot::open_absolute("/private/tmp")
  let result = @worker_process.prepare_with_approved_roots(
    executable,
    startup_contract(),
    bootstrap_source(),
    @worker_process.WorkerProcessLimits::new(
      max_frame_bytes=@worker_wire.bootstrap_source_capacity(),
      startup_io_timeout_millis=5000,
      io_timeout_millis=5000,
      shutdown_timeout_millis=5000,
    ),
    model,
    kernel,
  )
  model.close()
  kernel.close()
  match result {
    ReadyRootBoundProcess(owner) => owner
    RootBoundCleanupRequired(_) =>
      raise Failure("rooted worker startup cleanup required")
  }
}

///|
fn complete_exchange(
  owner : @worker_process.RootBoundWorkerProcessSupervisor,
) -> Unit raise {
  let mut attempts = 0
  while attempts < 1000000 {
    match owner.progress_exchange() {
      ExchangePending => attempts += 1
      ExchangeCompleted => return
    }
  }
  raise Failure("nonblocking worker exchange did not complete")
}

///|
fn run(executable : Bytes) -> Unit raise {
  let rooted = start_root_worker(executable)
  let rooted_plan = plan(1UL, 10UL)
  let exchange_sequence = rooted.begin_exchange(rooted_plan)
  assert_eq(exchange_sequence, 1UL)
  complete_exchange(rooted)
  let rooted_completion = rooted.received_for_sequence(exchange_sequence)
  assert_eq(rooted_completion.sequence_value(), 1UL)
  assert_eq(rooted_completion.frame(rooted).entry_count(), 1)
  rooted.retire_received(rooted_completion)
  rooted.begin_recovery()
  match rooted.restart() {
    RootBoundRestarted => ()
    RootBoundRestartCleanupRequired =>
      raise Failure("rooted exchange replacement cleanup required")
  }
  let replacement_plan = plan(2UL, 20UL)
  let replacement_sequence = rooted.begin_exchange(replacement_plan)
  assert_eq(replacement_sequence, 2UL)
  complete_exchange(rooted)
  let replacement_completion = rooted.received_for_sequence(
    replacement_sequence,
  )
  rooted.retire_received(replacement_completion)
  rooted.shutdown_clean()
  rooted.close()
  assert_true(rooted.lifecycle() == RootBoundClosed)

  let supervisor = start_worker(executable, startup_contract())
  assert_eq(supervisor.startup_contract(), startup_contract())
  let plan_a1 = plan(1UL, 11UL)
  let plan_b2 = plan(2UL, 12UL)
  let submission_a1 = supervisor.submit(plan_a1)
  let submission_b2 = supervisor.submit(plan_b2)
  let completion_a1 = supervisor.receive(submission_a1, plan_a1)
  let frame_a1 = completion_a1.frame(supervisor)
  assert_eq(frame_a1.entry_count(), 1)
  assert_eq(frame_a1.entry_for_slot(0).processed_tokens(), 2)
  supervisor.retire_received(completion_a1)
  let plan_a3 = plan(3UL, 13UL)
  let submission_a3 = supervisor.submit(plan_a3)
  let completion_b2 = supervisor.receive(submission_b2, plan_b2)
  supervisor.retire_received(completion_b2)
  let completion_a3 = supervisor.receive(submission_a3, plan_a3)
  supervisor.retire_received(completion_a3)
  assert_eq(supervisor.outstanding_count(), 0)
  supervisor.shutdown_clean()
  supervisor.close()
  assert_true(supervisor.lifecycle() == Closed)

  let graceful_restart = supervisor.recovery_startup_contract()
  assert_eq(graceful_restart.predecessor_value(), 3UL)
  let replacement = start_worker(executable, graceful_restart)
  let plan_b4 = plan(4UL, 14UL)
  let abandoned_b4 = replacement.submit(plan_b4)
  replacement.close()
  replacement.abandon_submission(abandoned_b4)

  let failed_restart = replacement.recovery_startup_contract()
  assert_eq(failed_restart.predecessor_value(), 4UL)
  let final_worker = start_worker(executable, failed_restart)
  let plan_a5 = plan(5UL, 15UL)
  let submission_a5 = final_worker.submit(plan_a5)
  let completion_a5 = final_worker.receive(submission_a5, plan_a5)
  assert_eq(completion_a5.frame(final_worker).entry_count(), 1)
  final_worker.retire_received(completion_a5)
  final_worker.shutdown_clean()
  final_worker.close()
  assert_eq(final_worker.recovery_startup_contract().predecessor_value(), 5UL)
}

///|
fn main {
  match @env.args() {
    [_, executable] => try! run(@utf8.encode(executable))
    _ => println("worker executable path is required")
  }
}

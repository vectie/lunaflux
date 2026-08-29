///|
fn supervisor_limits() -> @worker_protocol.WorkerProtocolLimits raise {
  @worker_protocol.WorkerProtocolLimits::new(
    max_prefill_rows=2,
    max_decode_rows=2,
    max_plan_rows=2,
    max_plan_tokens=8,
    max_plan_pages=8,
    max_capabilities=8,
    max_completion_slots=2,
    max_sequence_tokens=16,
    max_token_id=31,
  )
}

///|
fn supervisor_inference_limits() -> @inference.InferenceLimits raise {
  @inference.InferenceLimits::new(
    max_text_bytes=8,
    max_input_tokens=8,
    max_new_tokens=8,
    max_context_tokens=16,
    max_token_id=31,
    max_stop_token_ids=1,
    max_stop_strings=1,
    max_stop_string_bytes=1,
    max_trace_bytes=1,
    max_cache_scope_bytes=1,
    max_decoded_delta_bytes=1,
    max_deadline_millis=1000UL,
    max_top_k=31,
    max_temperature=4.0,
  )
}

///|
fn supervisor_process_limits() -> WorkerProcessLimits raise {
  WorkerProcessLimits::new(
    max_frame_bytes=@worker_wire.bootstrap_source_capacity(),
    startup_io_timeout_millis=1000,
    io_timeout_millis=1000,
    shutdown_timeout_millis=1000,
  )
}

///|
fn supervisor_source_for_roots(
  model_root : String,
  kernel_root : String,
) -> @worker_wire.EncodedBootstrapSource raise {
  @worker_wire.admit_bootstrap_source(
    @worker_wire.WorkerBootstrapRoots::new(model_root~, kernel_root~),
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
fn supervisor_source() -> @worker_wire.EncodedBootstrapSource raise {
  supervisor_source_for_roots("/models", "/kernels")
}

///|
fn alternate_supervisor_source() -> @worker_wire.EncodedBootstrapSource raise {
  let base = supervisor_source()
  @worker_wire.admit_bootstrap_source(
    @worker_wire.WorkerBootstrapRoots::new(
      model_root="/alternate-models",
      kernel_root="/kernels",
    ),
    base.model(),
    base.kernel(),
    base.execution(),
    base.ceilings(),
  )
}

///|
fn supervisor_startup_for_source(
  source : @worker_wire.EncodedBootstrapSource,
) -> @worker_wire.WorkerStartupContract raise {
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
    bootstrap_source_digest=source.digest(),
    device_ordinal=0,
    model_generation_value=1UL,
    worker_limits=supervisor_limits(),
    inference_limits=supervisor_inference_limits(),
    has_predecessor=false,
    predecessor_value=0UL,
  )
}

///|
fn supervisor_startup() -> @worker_wire.WorkerStartupContract raise {
  supervisor_startup_for_source(supervisor_source())
}

///|
fn test_supervisor(
  executable? : Bytes = @worker_executable_fixture.cat_path(),
  shutdown_timeout_millis? : Int = 1000,
) -> WorkerProcessSupervisor raise {
  let worker_limits = supervisor_limits()
  let inference_limits = supervisor_inference_limits()
  let plan_capacity = @worker_wire.plan_frame_byte_capacity(worker_limits)
  let tx_a = FixedArray::make(plan_capacity, b'\x00')
  let tx_b = FixedArray::make(plan_capacity, b'\x00')
  let plan_a = @worker_wire.PlanFrameBuffer::new_transport_owned(
    worker_limits,
    inference_limits,
    tx_a,
    frame_offset=0,
  )
  let plan_b = @worker_wire.PlanFrameBuffer::new_transport_owned(
    worker_limits,
    inference_limits,
    tx_b,
    frame_offset=0,
  )
  let completion_a = @worker_wire.CompletionFrameBuffer::new(
    worker_limits, inference_limits,
  )
  let completion_b = @worker_wire.CompletionFrameBuffer::new(
    worker_limits, inference_limits,
  )
  let limits = supervisor_process_limits()
  {
    child: spawn_worker_fixture_child(
      executable,
      @process.ProcessLimits::new(
        max_frame_bytes=limits.max_frame_bytes(),
        io_timeout_millis=limits.io_timeout_millis(),
        shutdown_timeout_millis~,
      ),
    ),
    child_selected: true,
    startup: supervisor_startup(),
    source: supervisor_source(),
    parent_approval: None,
    plan_a,
    plan_b,
    completion_a,
    completion_b,
    tx_a,
    tx_b,
    rx: FixedArray::make(completion_a.capacity(), b'\x00'),
    rx_staging: FixedArray::make(completion_a.capacity(), b'\x00'),
    graph_rx: FixedArray::make(
      @worker_wire.worker_graph_telemetry_frame_capacity(),
      b'\x00',
    ),
    graph_rx_staging: FixedArray::make(
      @worker_wire.worker_graph_telemetry_frame_capacity(),
      b'\x00',
    ),
    state_a: 0,
    state_b: 0,
    sequence_a: 0UL,
    sequence_b: 0UL,
    graph_telemetry_a: @worker_wire.WorkerGraphTelemetrySlot::new(),
    graph_telemetry_b: @worker_wire.WorkerGraphTelemetrySlot::new(),
    last_graph_telemetry: @worker_wire.WorkerGraphTelemetrySlot::new(),
    last_submitted: 0UL,
    last_retired: 0UL,
    maintenance_phase: 0,
    maintenance_failure: None,
    lifecycle: Ready,
  }
}

///|
fn supervisor_page() -> @page_allocator.PageId raise {
  @page_allocator.PageAllocator::new(
    @page_allocator.PageAllocatorLimits::new(
      page_capacity=2,
      max_run_pages=2,
      max_active_references_per_page=2,
      max_cached_references_per_page=2,
    ),
  ).allocate_one()
}

///|
fn supervisor_plan(
  sequence : UInt64,
  generation? : UInt64 = 1UL,
  completion_slot? : Int = 0,
) -> @worker_protocol.SubmittedSchedulePlan raise {
  let limits = supervisor_limits()
  let owner = @worker_protocol.SchedulePlanBuffer::new(limits)
  owner.append_prefill(
    request_id=@inference.RequestId::new(sequence),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_start=0,
    tokens=[1, 2],
    pages=[supervisor_page()],
    capabilities=[@plan.KernelCapabilityId::embedding_lookup()],
    completion_slot=@worker_protocol.CompletionSlot::new(
      completion_slot, limits,
    ),
  )
  let generation = @worker_protocol.ModelPlanGeneration::from_wire(generation)
  owner.submit(
    sequence=@worker_protocol.PlanSequence::from_wire(sequence),
    model_plan_generation=generation,
    token_budget=2,
    if sequence == 1UL {
      None
    } else {
      Some(@worker_protocol.PlanSequence::from_wire(sequence - 1UL))
    },
    generation,
  )
}

///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
test "submission rejects a foreign loaded model generation before write" {
  let owner = test_supervisor()
  try owner.submit(supervisor_plan(1UL, generation=2UL)) catch {
    Invalid(Submission) => ()
    _ => fail("unexpected foreign-generation error")
  } noraise {
    _ => fail("foreign model generation reached the child")
  }
  assert_eq(owner.outstanding_count(), 0)
  owner.close()
}

///|
fn encoded_completion(
  plan : @worker_protocol.SubmittedSchedulePlan,
  destination : FixedArray[Byte],
) -> Int raise {
  let limits = supervisor_limits()
  let inference = supervisor_inference_limits()
  let plan_owner = @worker_wire.PlanFrameBuffer::new(limits, inference)
  let frame = plan_owner.encode_plan(plan)
  let completion_owner = @worker_wire.CompletionFrameBuffer::new(
    limits, inference,
  )
  let writer = completion_owner.writer_for(frame)
  writer.append_prefill_completed(frame.prefill_row(0))
  let completion = writer.submit()
  completion.copy_to(destination, destination_offset=0)
  completion.length()
}

///|
fn encoded_graph_telemetry(
  sequence : UInt64,
  hits : UInt64,
  destination : FixedArray[Byte],
) -> Int raise {
  let shape = @worker_wire.WorkerGraphShapeClass::new(
    phase=1,
    batch_rows=1,
    query_tokens_per_row=1,
    context_tokens=1,
    token_rows=1L,
  )
  @worker_wire.encode_worker_graph_telemetry(
    @worker_wire.WorkerGraphTelemetry::with_shape(
      sequence~,
      path=WorkerGraphCaptured,
      shape~,
      graph_hits=hits,
      graph_misses=0UL,
    ),
    destination,
  )
}

///|
fn inject_completion(
  owner : WorkerProcessSupervisor,
  side : Int,
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> ReceivedWorkerCompletion raise {
  let length = encoded_completion(plan, owner.rx)
  owner.accept_completion_bytes_in_place(side, plan, length)
  let sequence = plan.sequence_wire()
  let graph_length = encoded_graph_telemetry(sequence, sequence, owner.graph_rx)
  owner.accept_graph_telemetry_bytes_in_place(side, graph_length)
  ReceivedWorkerCompletion(owner, side, sequence)
}

///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
test "received side remains pinned until explicit retirement" {
  let owner = test_supervisor()
  let plan = supervisor_plan(1UL)
  let submitted = owner.submit(plan)
  assert_eq(submitted.sequence_value(), 1UL)
  let received = inject_completion(owner, 0, plan)
  let length = received.frame(owner).length()
  assert_eq(owner.outstanding_count(), 1)
  assert_eq(received.frame(owner).length(), length)
  let graph = received.graph_telemetry(owner)
  assert_eq(graph.sequence_value(), 1UL)
  assert_eq(graph.selected_path(), WorkerGraphCaptured)
  assert_eq(graph.graph_hits(), 1UL)
  assert_true(graph.has_selected_shape_class())
  assert_eq(received.frame(owner).length(), length)
  owner.retire_received(received)
  assert_eq(owner.outstanding_count(), 0)
  try received.frame(owner) catch {
    Invalid(Completion) => ()
    _ => fail("unexpected retired-frame error")
  } noraise {
    _ => fail("retired completion remained live")
  }
  owner.close()
}

///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
test "A side reuses only after ordered A/B retirement" {
  let owner = test_supervisor()
  let plan_a1 = supervisor_plan(1UL)
  let plan_b2 = supervisor_plan(2UL)
  let submission_a1 = owner.submit(plan_a1)
  let submission_b2 = owner.submit(plan_b2)
  let received_a1 = inject_completion(owner, 0, plan_a1)
  try owner.receive(submission_b2, plan_b2) catch {
    Backpressured => ()
    _ => fail("unexpected pinned-A backpressure error")
  } noraise {
    _ => fail("B response was read while A remained pinned")
  }
  owner.retire_received(received_a1)
  let plan_a3 = supervisor_plan(3UL)
  let submission_a3 = owner.submit(plan_a3)
  assert_eq(submission_a1.sequence_value(), 1UL)
  assert_eq(submission_a3.sequence_value(), 3UL)
  let received_b2 = inject_completion(owner, 1, plan_b2)
  owner.retire_received(received_b2)
  let received_a3 = inject_completion(owner, 0, plan_a3)
  owner.retire_received(received_a3)
  assert_eq(owner.outstanding_count(), 0)
  owner.close()
}

///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
test "A/B pressure and out-of-order receive fail before channel read" {
  let owner = test_supervisor()
  let plan_a = supervisor_plan(1UL)
  let plan_b = supervisor_plan(2UL)
  let a = owner.submit(plan_a)
  let b = owner.submit(plan_b)
  assert_false(owner.has_capacity())
  try owner.receive(b, plan_b) catch {
    Invalid(Sequence) => ()
    _ => fail("unexpected out-of-order response error")
  } noraise {
    _ => fail("out-of-order response reached the channel")
  }
  try owner.submit(supervisor_plan(3UL)) catch {
    Backpressured => ()
    _ => fail("unexpected A/B pressure error")
  } noraise {
    _ => fail("third outstanding plan was admitted")
  }
  assert_eq(a.sequence_value(), 1UL)
  owner.close()
}

///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
test "malformed child response permanently fails the session" {
  let owner = test_supervisor()
  let plan = supervisor_plan(1UL)
  let submission = owner.submit(plan)
  try owner.receive(submission, plan) catch {
    Transport(InvalidConfiguration) => ()
    error => fail("unexpected echoed-plan response error: \{Repr(error)}")
  } noraise {
    _ => fail("plan frame was accepted as a completion")
  }
  assert_true(owner.lifecycle() == Failed)
  owner.close()
}

///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
test "closed submissions are abandoned only in exact sequence order" {
  let owner = test_supervisor()
  let plan_a = supervisor_plan(1UL)
  let plan_b = supervisor_plan(2UL)
  let submission_a = owner.submit(plan_a)
  let submission_b = owner.submit(plan_b)
  try owner.recovery_startup_contract() catch {
    Invalid(Recovery) => ()
    _ => fail("unexpected live recovery-contract error")
  } noraise {
    _ => fail("live child published a recovery contract")
  }
  owner.close()
  try owner.abandon_submission(submission_b) catch {
    Invalid(Sequence) => ()
    _ => fail("unexpected out-of-order abandonment error")
  } noraise {
    _ => fail("later submission was abandoned first")
  }
  owner.abandon_submission(submission_a)
  try owner.recovery_startup_contract() catch {
    Invalid(Recovery) => ()
    _ => fail("unexpected outstanding recovery-contract error")
  } noraise {
    _ => fail("outstanding submission was omitted from recovery")
  }
  owner.abandon_submission(submission_b)
  let recovery = owner.recovery_startup_contract()
  assert_eq(recovery.identity(), supervisor_startup().identity())
  assert_eq(
    recovery.bootstrap_digest(),
    supervisor_startup().bootstrap_digest(),
  )
  assert_eq(
    recovery.bootstrap_source_digest(),
    supervisor_startup().bootstrap_source_digest(),
  )
  assert_eq(recovery.model_generation_value(), 1UL)
  assert_eq(recovery.device_ordinal(), supervisor_startup().device_ordinal())
  assert_true(recovery.has_predecessor())
  assert_eq(recovery.predecessor_value(), 2UL)
  assert_eq(owner.outstanding_count(), 0)
}

///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
test "validated completion remains retireable after child close" {
  let owner = test_supervisor()
  let plan = supervisor_plan(1UL)
  ignore(owner.submit(plan))
  let received = inject_completion(owner, 0, plan)
  owner.close()
  assert_eq(received.frame(owner).entry_count(), 1)
  owner.retire_received(received)
  let recovery = owner.recovery_startup_contract()
  assert_true(recovery.has_predecessor())
  assert_eq(recovery.predecessor_value(), 1UL)
  assert_eq(owner.outstanding_count(), 0)
}

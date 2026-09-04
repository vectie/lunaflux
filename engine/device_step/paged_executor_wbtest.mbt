///|
priv struct FakePagedStep {
  operation : @plan.OperationId
}

///|
fn[Step] run_paged_ordered_step_range(
  steps : ReadOnlyArray[Step],
  lifecycle : PagedGraphLifecycle,
  set_lifecycle : (PagedGraphLifecycle) -> Unit,
  enqueue : (Int, Step) -> Unit raise DeviceStepError,
  record : () -> Unit raise DeviceStepError,
  wait : () -> Unit raise DeviceStepError,
  abort : () -> Unit,
) -> Unit raise DeviceStepError {
  require_paged_lifecycle(lifecycle, GraphStaged)
  set_lifecycle(GraphExecuting)
  let mut execution_complete = false
  defer (if !execution_complete {
    set_lifecycle(GraphPoisoned)
    abort()
  })
  for index, step in steps {
    enqueue(index, step)
  }
  record()
  wait()
  set_lifecycle(GraphExecuted)
  execution_complete = true
}

///|
fn dummy_paged_executor(
  lifecycle : PagedGraphLifecycle,
  fp8_frame_admission? : Fp8FrameAdmissionStateV3? = None,
) -> PagedGraphExecutor {
  let limits = try! @sampling.SamplingLimits::new(1, 1)
  let profile = try! @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=1,
    max_query_tokens=1,
    max_page_table_entries=1,
  )
  let graph_bucket_table = new_paged_graph_bucket_table(profile)
  {
    resources: empty_paged_resources(),
    completion: {
      final_output_offset: 0L,
      row_bytes: 2L,
      vocabulary_size: 1,
      limits,
      sampler: try! @sampling.SamplerScratch::new(1, limits),
      encoded_logits: FixedArray::make(2, b'\x00'),
      logits: FixedArray::make(1, 0.0),
      selected_tokens: FixedArray::make(1, 0),
      mode: HostLogitsCompletion,
    },
    fp8_frame_envelope: None,
    fp8_frame_admission,
    fp8_scale_evidence: None,
    graph_telemetry: new_paged_graph_telemetry(OrderedKernelEager, None),
    graph_bucket_table,
    ordered_dispatch: {
      slot_owner_indices: FixedArray::make(
        graph_bucket_table.bucket_count(),
        -1,
      ),
      prefill_split_slot_owner_indices: FixedArray::make(
        graph_bucket_table.bucket_count(),
        -1,
      ),
      fallback_owner_index: 0,
      telemetry_owner_index: 0,
      prefill_split_owner_index: -1,
      prefill_split_minimum_context_tokens: PAGED_DECODE_SPLIT_DISABLED_CONTEXT_THRESHOLD,
      decode_owner_index: -1,
      decode_split_owner_index: -1,
      decode_split_minimum_context_tokens: PAGED_DECODE_SPLIT_DISABLED_CONTEXT_THRESHOLD,
    },
    fused_residual_spans: ReadOnlyArray::from_array([]),
    fused_dispatch_canary: None,
    lifecycle,
  }
}

///|
fn executor_test_submitted_plan() -> @worker_protocol.SubmittedSchedulePlan raise {
  let allocator = step_allocator()
  let buffer = @worker_protocol.SchedulePlanBuffer::new(step_protocol_limits())
  buffer.append_prefill(
    request_id=step_request(1UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_start=0,
    tokens=[1],
    pages=[allocator.allocate_one()],
    capabilities=step_required_capability_array(),
    completion_slot=step_slot(0),
  )
  submit_step_buffer(buffer, token_budget=1)
}

///|
fn executor_test_stochastic_plan() -> @worker_protocol.SubmittedSchedulePlan raise {
  let allocator = step_allocator()
  let buffer = @worker_protocol.SchedulePlanBuffer::new(step_protocol_limits())
  let inference_limits = @inference.InferenceLimits::new(
    max_text_bytes=16,
    max_input_tokens=16,
    max_new_tokens=16,
    max_context_tokens=16,
    max_token_id=64,
    max_stop_token_ids=1,
    max_stop_strings=1,
    max_stop_string_bytes=1,
    max_trace_bytes=1,
    max_cache_scope_bytes=1,
    max_decoded_delta_bytes=1,
    max_deadline_millis=1UL,
    max_top_k=64,
    max_temperature=4.0,
  )
  buffer.append_final_prefill(
    request_id=step_request(2UL),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_start=0,
    tokens=[1],
    pages=[allocator.allocate_one()],
    capabilities=step_required_capability_array(),
    sampling=@inference.SamplingParameters::sample(
      temperature=1.0,
      top_k=2,
      inference_limits,
    ),
    sampling_seed=step_sampling_seed(),
    sample_index=0,
    completion_slot=step_slot(0),
  )
  submit_step_buffer(buffer, token_budget=1)
}

///|
test "submitted schedules select a phase-specific graph bucket before staging" {
  let profile = @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(77),
    max_query_rows=8,
    max_query_tokens=32,
    max_page_table_entries=64,
  )
  let bucket = select_submitted_graph_bucket(
    new_paged_graph_bucket_table(profile),
    executor_test_submitted_plan(),
  )
  assert_eq(bucket.phase(), PrefillGraph)
  assert_eq(bucket.upper_bounds(), (1, 1, 1))
}

///|
test "CUDA greedy startup mode rejects stochastic work before staging" {
  let limits = step_device_limits()
  try
    preflight_and_write(
      executor_test_stochastic_plan(),
      1UL,
      step_test_identity(),
      step_test_layout(),
      limits,
      make_host_buffers(limits),
      step_required_capabilities(),
      false,
      greedy_only=true,
    )
  catch {
    InvalidExecutor(SamplingMode) => ()
    _ => fail("unexpected CUDA-greedy sampling-mode failure")
  } noraise {
    _ => fail("greedy-only preflight silently accepted stochastic work")
  }
}

///|
test "CUDA greedy startup admission rejects an arbitrary offline module" {
  let metadata = blueprint_metadata()
  let model = blueprint_model_plan(metadata)
  let sampling_model = @llama.build_paged(metadata)
  let layout = blueprint_layout(model)
  let target = blueprint_target()
  let static_plan = full_static_plan(metadata, model, layout, target)
  let memory = blueprint_memory_plan(metadata, model, layout, target)
  let contracts = full_contract_set(model, static_plan, memory, layout, target)
  let artifacts = blueprint_artifacts(contracts)
  let compiler = @luna_specialization.LunaCompilerPolicy::new(
    identity=@luna_specialization.LunaCompilerIdentity::new(
      toolchain_digest_sha256="2".repeat(64),
      major=13,
      minor=0,
      patch=0,
    ),
    optimization_level=3,
    fused_multiply_add=false,
    reassociate=false,
    max_registers=64,
  )
  let limits = @luna_cuda_sampling_aot.GreedySamplingAotLimits::new(
    max_source_bytes=16384,
    max_recipe_bytes=4096,
    max_module_bytes=65536L,
  )
  let lowering = @luna_cuda_sampling_aot.lower_greedy_sampling_cuda_aot(
    @luna_cuda_sampling_aot.admit_host_sampling_boundary(sampling_model),
    target,
    compiler,
    blueprint_limits().max_rows(),
    limits,
  )
  let arbitrary = b"caller-selected-unmanifested-cubin"
  let digest = @crypto.bytes_to_hex_string(@crypto.sha256(arbitrary))
  let receipt = @luna_cuda_sampling_aot.GreedySamplingCompileReceipt::from_offline_builder(
    source_sha256=lowering.source_sha256(),
    recipe_sha256=lowering.recipe_sha256(),
    toolchain_sha256="2".repeat(64),
    driver_sha256="3".repeat(64),
    first_module_sha256=digest,
    second_module_sha256=digest,
  )
  let artifact = @luna_cuda_sampling_aot.admit_greedy_sampling_cuda_artifact(
    lowering, arbitrary, arbitrary, receipt, limits,
  )
  try
    admit_paged_cuda_greedy_sampling(sampling_model, artifacts, artifact)
  catch {
    InvalidExecutor(SamplingArtifact) => ()
    _ => fail("unexpected unmanifested-sampler rejection")
  } noraise {
    _ => fail("unmanifested arbitrary sampling module acquired load authority")
  }
}

///|
test "embedded CUDA greedy admission binds the exact LM-head module" {
  let metadata = blueprint_metadata()
  let model = blueprint_model_plan(metadata, final_language_model_head=true)
  let layout = blueprint_layout(model)
  let target = blueprint_target()
  let static_plan = full_static_plan(metadata, model, layout, target)
  let memory = blueprint_memory_plan(metadata, model, layout, target)
  let contracts = full_contract_set(model, static_plan, memory, layout, target)
  let artifacts = blueprint_artifacts(contracts)
  let blueprint = admit_paged_graph_execution_blueprint(
    model,
    blueprint_limits(),
    layout,
    static_plan,
    memory,
    blueprint_profile().id(),
    contracts,
    artifacts,
  )
  let policy = admit_paged_embedded_cuda_greedy_sampling(
    model, blueprint, artifacts,
  )
  guard policy is PagedCudaGreedySampling(admission) else {
    fail("embedded reducer did not select CUDA greedy mode")
  }
  assert_eq(
    admission.function_symbol,
    @luna_cuda_sampling_aot.greedy_sampling_cuda_function_symbol(),
  )
  assert_eq(admission.vocabulary_size, 8)
  assert_eq(admission.maximum_rows, blueprint_limits().max_rows())
  validate_paged_sampling_execution_policy(model, blueprint, artifacts, policy)
}

///|
test "public paged capabilities reject lifecycle and foreign owners" {
  let closed = dummy_paged_executor(GraphClosed)
  try closed.stage(executor_test_submitted_plan()) catch {
    InvalidExecutorLifecycle(expected=GraphReady, actual=GraphClosed) => ()
    _ => fail("unexpected closed stage error")
  } noraise {
    _ => fail("closed executor admitted a stage")
  }
  let first = dummy_paged_executor(GraphStaged)
  let second = dummy_paged_executor(GraphStaged)
  let before = second.graph_runtime_telemetry()
  let table = new_paged_graph_bucket_table(
    try! @launch_contract.PagedKernelProfile::new(
      id=@launch_contract.KernelProfileId::from_stable_int(1),
      max_query_rows=1,
      max_query_tokens=1,
      max_page_table_entries=1,
    ),
  )
  let graph_bucket = try! table.select(
    try! @luna_execution_graph_strategy.ExecutionGraphShape::new(
      phase=PrefillGraph,
      batch_rows=1,
      query_tokens=1,
      context_tokens=1,
    ),
  )
  let staged : StagedPagedGraphExecution = {
    executor: first,
    epoch: 1UL,
    graph_bucket,
    ordered_executor_index: 0,
  }
  try second.execute(staged) catch {
    ExecutorOwnerMismatch => ()
    _ => fail("unexpected foreign staged capability error")
  } noraise {
    _ => fail("foreign staged capability was admitted")
  }
  let after = second.graph_runtime_telemetry()
  assert_eq(after.selected_path(), before.selected_path())
  assert_eq(after.graph_hits(), before.graph_hits())
  assert_eq(after.graph_misses(), before.graph_misses())
  first.lifecycle = GraphExecuted
  second.lifecycle = GraphExecuted
  let executed : ExecutedPagedGraphExecution = {
    executor: first,
    epoch: 1UL,
    ordered_executor_index: 0,
  }
  try second.finish(executed) catch {
    ExecutorOwnerMismatch => ()
    _ => fail("unexpected foreign executed capability error")
  } noraise {
    _ => fail("foreign executed capability was admitted")
  }
  let missing = dummy_paged_executor(GraphExecuted)
  assert_eq(
    closed.graph_runtime_telemetry().selected_path(),
    PagedGraphOrderedEager,
  )
  let missing_capability : ExecutedPagedGraphExecution = {
    executor: missing,
    epoch: 1UL,
    ordered_executor_index: 0,
  }
  let plan = executor_test_submitted_plan()
  let writer = @worker_protocol.CompletionBuffer::new(step_protocol_limits()).writer_for(
    plan,
  )
  try missing.sample_completion(missing_capability, plan, writer) catch {
    InvalidExecutor(ResourceState) => ()
    _ => fail("unexpected missing completion-resource error")
  } noraise {
    _ => fail("completion sampled without executor resources")
  }
}

///|
test "ordered paged execution records and waits once after semantic enqueue" {
  let lifecycle = Ref(GraphStaged)
  let actions : Array[String] = []
  let steps = ReadOnlyArray::from_array([
    { operation: @plan.OperationId::new(0) },
    { operation: @plan.OperationId::new(1) },
    { operation: @plan.OperationId::new(2) },
  ])
  run_paged_ordered_step_range(
    steps,
    lifecycle.val,
    state => lifecycle.val = state,
    (index, step) => actions.push("enqueue:\{index}:\{step.operation.as_int()}"),
    () => actions.push("record"),
    () => actions.push("wait"),
    () => actions.push("abort"),
  )
  assert_eq(actions, [
    "enqueue:0:0", "enqueue:1:1", "enqueue:2:2", "record", "wait",
  ])
  assert_eq(lifecycle.val, GraphExecuted)
}

///|
test "ordered paged execution drains and poisons after partial enqueue" {
  let lifecycle = Ref(GraphStaged)
  let actions : Array[String] = []
  let steps = ReadOnlyArray::from_array([
    { operation: @plan.OperationId::new(0) },
    { operation: @plan.OperationId::new(1) },
    { operation: @plan.OperationId::new(2) },
  ])
  try
    run_paged_ordered_step_range(
      steps,
      lifecycle.val,
      state => lifecycle.val = state,
      (index, step) => {
        actions.push("enqueue:\{index}")
        if step.operation.as_int() == 1 {
          raise ExecutorLaunchFailed(
            step.operation,
            DriverFailure(OrderedExecutorEnqueue),
          )
        }
      },
      () => actions.push("record"),
      () => actions.push("wait"),
      () => actions.push("abort"),
    )
  catch {
    ExecutorLaunchFailed(operation, DriverFailure(OrderedExecutorEnqueue)) =>
      assert_eq(operation.as_int(), 1)
    _ => fail("unexpected ordered paged launch failure")
  } noraise {
    _ => fail("injected ordered launch failure was swallowed")
  }
  assert_eq(actions, ["enqueue:0", "enqueue:1", "abort"])
  assert_eq(lifecycle.val, GraphPoisoned)
}

///|
test "paged graph policy admits eager fallback only by explicit variant" {
  assert_eq(
    paged_ordered_device_policy(PagedOrderedEagerOnly),
    OrderedKernelEagerOnly,
  )
  assert_eq(
    paged_ordered_device_policy(PagedCapturedRequired),
    OrderedKernelCaptureRequired,
  )
  assert_eq(
    paged_ordered_device_policy(PagedCapturedWithEagerFallback),
    OrderedKernelCaptureWithEagerFallback,
  )
}

///|
test "full blueprint builds one exact function map per entry point" {
  let metadata = blueprint_metadata()
  let plan = blueprint_model_plan(metadata)
  let layout = blueprint_layout(plan)
  let target = blueprint_target()
  let static_plan = full_static_plan(metadata, plan, layout, target)
  let memory = blueprint_memory_plan(metadata, plan, layout, target)
  let contracts = full_contract_set(plan, static_plan, memory, layout, target)
  let blueprint = admit_paged_graph_execution_blueprint(
    plan,
    blueprint_limits(),
    layout,
    static_plan,
    memory,
    blueprint_profile().id(),
    contracts,
    blueprint_artifacts(contracts),
  )
  let (functions, indices, split_decode_indices) = collect_paged_functions(
    blueprint,
  )
  assert_eq(functions.length(), blueprint.steps().length())
  assert_eq(indices.length(), blueprint.steps().length())
  assert_eq(split_decode_indices.length(), blueprint.steps().length())
  for index, function in functions {
    assert_eq(indices[index], index)
    assert_eq(function.identity, blueprint.steps()[index].entry_point())
    assert_eq(function.module_index, 0)
    assert_eq(split_decode_indices[index], -1)
  }
}

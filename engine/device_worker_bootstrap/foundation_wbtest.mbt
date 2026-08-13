///|
fn bootstrap_test_startup(
  source_digest : @worker_wire.WorkerBootstrapSourceDigest,
  content? : String = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  inference_context? : Int = 16,
  inference_token_id? : Int = 7,
) -> @worker_wire.WorkerStartupContract raise {
  let identity = @spec.ModelIdentity::new(
    @spec.ContentDigest::from_sha256(content),
    @spec.PlanDigest::from_sha256(
      "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    ),
  )
  @worker_wire.WorkerStartupContract::new(
    identity~,
    bootstrap_digest=@worker_wire.WorkerBootstrapDigest::from_sha256(
      "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
    ),
    bootstrap_source_digest=source_digest,
    device_ordinal=0,
    model_generation_value=1UL,
    worker_limits=@worker_protocol.WorkerProtocolLimits::new(
      max_prefill_rows=1,
      max_decode_rows=1,
      max_plan_rows=1,
      max_plan_tokens=4,
      max_plan_pages=2,
      max_capabilities=64,
      max_completion_slots=1,
      max_sequence_tokens=16,
      max_token_id=7,
    ),
    inference_limits=@inference.InferenceLimits::new(
      max_text_bytes=32,
      max_input_tokens=inference_context,
      max_new_tokens=inference_context,
      max_context_tokens=inference_context,
      max_token_id=inference_token_id,
      max_stop_token_ids=1,
      max_stop_strings=1,
      max_stop_string_bytes=8,
      max_trace_bytes=8,
      max_cache_scope_bytes=8,
      max_decoded_delta_bytes=8,
      max_deadline_millis=100UL,
      max_top_k=inference_token_id + 1,
      max_temperature=4.0,
    ),
    has_predecessor=false,
    predecessor_value=0UL,
  )
}

///|
fn bootstrap_test_source() -> @worker_wire.EncodedBootstrapSource raise {
  @worker_wire.admit_bootstrap_source(
    @worker_wire.WorkerBootstrapRoots::new(
      model_root="/model",
      kernel_root="/kernel",
    ),
    @worker_wire.WorkerBootstrapModelSource::new(
      config_locator="config.json",
      weights_locator="weights.safetensors",
      config_digest_sha256="dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
    ),
    @worker_wire.WorkerBootstrapKernelSource::new(
      manifest_locator="execution.json",
      manifest_digest_sha256="eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
    ),
    @worker_wire.WorkerBootstrapExecution::new(
      compute_major=8,
      compute_minor=0,
      supports_bf16=true,
      supports_cublas_lt=false,
      tokens_per_page=4,
      total_page_count=8,
    ),
    @worker_wire.WorkerBootstrapCeilings::new(
      max_model_config_bytes=4096,
      max_weight_file_bytes=4096L,
      max_weight_arena_bytes=4096L,
      max_activation_arena_bytes=65536L,
      max_kv_arena_bytes=65536L,
      max_execution_manifest_bytes=65536L,
      max_module_bytes=4096L,
      max_total_module_bytes=4096L,
    ),
  )
}

///|
test "source mismatch fails before root acquisition" {
  let source = bootstrap_test_source()
  let startup = bootstrap_test_startup(
    @worker_wire.WorkerBootstrapSourceDigest::from_sha256(
      "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
    ),
  )
  let acquisitions = Ref(0)
  try {
    validate_source_envelope(startup, source)
    ignore(
      acquire_roots_with(_role => {
        acquisitions.val += 1
        raise Root(role=ModelRoot, issue=Unavailable)
      }),
    )
  } catch {
    Invalid(SourceDigest) => ()
    _ => fail("unexpected source mismatch error")
  } noraise {
    _ => fail("source mismatch was accepted")
  }
  assert_eq(acquisitions.val, 0)
}

///|
async test "partial role acquisition closes the model root" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-bootstrap-root-partial")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let model = @approved_fs.ApprovedRoot::open_absolute(
      @fs.realpath(directory),
    )
    try
      acquire_roots_with(role => {
        match role {
          ModelRoot => model
          KernelRoot => raise Root(role=KernelRoot, issue=Unavailable)
        }
      })
    catch {
      Root(role=KernelRoot, issue=Unavailable) => ()
      _ => fail("unexpected partial acquisition error")
    } noraise {
      _ => fail("partial role acquisition succeeded")
    }
    assert_true(model.is_closed())
  }
}

///|
test "exact graph ceiling rejects before plan allocation" {
  assert_eq(exact_graph_count(1), 12)
  assert_eq(exact_graph_count(454), 4089)
  try exact_graph_count(455) catch {
    Invalid(OperationCount) => ()
    _ => fail("unexpected graph ceiling error")
  } noraise {
    _ => fail("oversized graph count was accepted")
  }
}

///|
test "narrower inference envelope remains valid" {
  let source = bootstrap_test_source()
  let startup = bootstrap_test_startup(
    source.digest(),
    inference_context=8,
    inference_token_id=3,
  )
  let model = @spec.LlamaModelMetadata::from_verified_content(
    @spec.LlamaModelSpec::new(
      vocabulary_size=8,
      hidden_size=4,
      intermediate_size=6,
      layer_count=1,
      attention_head_count=2,
      kv_head_count=1,
      context_length=16,
      rope_theta=10000.0,
      weight_dtype=BFloat16,
      kv_dtype=BFloat16,
    ),
    startup.identity().content(),
  )
  validate_runtime_envelope(startup, model, source.execution())
}

///|
test "cleanup retries independent worker model and kernel authorities" {
  let roots : BootstrapRoots = { model: None, kernel: None }
  let failed : FailedDeviceWorkerBootstrapPreparation = {
    roots,
    worker: None,
    cause: Invalid(ReadinessContract),
    worker_cleanup_failure: Some(CleanupWorker(Invalid(StartupContract))),
    model_cleanup_failure: Some(CleanupRoot(role=ModelRoot, issue=Busy)),
    kernel_cleanup_failure: Some(CleanupRoot(role=KernelRoot, issue=Failed)),
    closed: false,
  }
  failed.retry_close()
  assert_true(failed.is_closed())
  assert_eq(failed.cleanup_failure(), None)
}

///|
test "compound cleanup probes every independent authority" {
  let events : Array[String] = []
  let failures = cleanup_all_with(
    () => {
      events.push("worker")
      raise CleanupWorker(Invalid(StartupContract))
    },
    () => {
      events.push("model")
      raise CleanupRoot(role=ModelRoot, issue=Busy)
    },
    () => {
      events.push("kernel")
      raise CleanupRoot(role=KernelRoot, issue=Failed)
    },
  )
  assert_eq(events, ["worker", "model", "kernel"])
  assert_eq(failures.0, Some(CleanupWorker(Invalid(StartupContract))))
  assert_eq(failures.1, Some(CleanupRoot(role=ModelRoot, issue=Busy)))
  assert_eq(failures.2, Some(CleanupRoot(role=KernelRoot, issue=Failed)))
}

///|
test "weight header is independent of config bytes and admits arena padding" {
  let ceilings = @worker_wire.WorkerBootstrapCeilings::new(
    max_model_config_bytes=4096,
    max_weight_file_bytes=4096L,
    max_weight_arena_bytes=8192L,
    max_activation_arena_bytes=65536L,
    max_kv_arena_bytes=65536L,
    max_execution_manifest_bytes=65536L,
    max_module_bytes=4096L,
    max_total_module_bytes=4096L,
  )
  let limits = weight_header_limits(ceilings, 12)
  assert_eq(limits.max_header_bytes(), RECIPE_MAX_WEIGHT_HEADER_BYTES)
  assert_eq(limits.max_payload_bytes(), 4096L)
  assert_eq(limits.max_materialized_tensor_bytes(), 4096L)
}

///|
fn orchestration_fixture(
  fail_kernel? : Bool = false,
  fail_model? : Bool = false,
  mismatch? : Bool = false,
) -> (Array[String], DeviceWorkerBootstrapPreparation) raise {
  let source = bootstrap_test_source()
  let startup = bootstrap_test_startup(source.digest())
  let events : Array[String] = []
  let result = orchestrate_worker_transition(
    startup,
    () => {
      events.push("kernel-close")
      if fail_kernel {
        raise CleanupRoot(role=KernelRoot, issue=Busy)
      }
    },
    () => {
      events.push("worker-prepare")
      7
    },
    _worker => None,
    () => {
      events.push("model-close")
      if fail_model {
        raise CleanupRoot(role=ModelRoot, issue=Busy)
      }
    },
    _worker => {
      events.push("readiness")
      if mismatch {
        try! bootstrap_test_startup(
          source.digest(),
          content="ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        )
      } else {
        startup
      }
    },
    (worker, error) => {
      if worker is Some(_) {
        events.push("worker-cleanup")
      }
      CleanupRequired({
        roots: { model: None, kernel: None },
        worker: None,
        cause: error,
        worker_cleanup_failure: None,
        model_cleanup_failure: None,
        kernel_cleanup_failure: None,
        closed: false,
      })
    },
    _worker => {
      events.push("ready")
      CleanupRequired({
        roots: { model: None, kernel: None },
        worker: None,
        cause: Invalid(ReadinessContract),
        worker_cleanup_failure: None,
        model_cleanup_failure: None,
        kernel_cleanup_failure: None,
        closed: false,
      })
    },
  )
  (events, result)
}

///|
test "orchestration closes roots in order before readiness" {
  let (events, _) = orchestration_fixture()
  assert_eq(events, [
    "kernel-close", "worker-prepare", "model-close", "readiness", "ready",
  ])
}

///|
test "kernel close blocks device preparation and retains root failure" {
  let (events, result) = orchestration_fixture(fail_kernel=true)
  assert_eq(events, ["kernel-close"])
  guard result is CleanupRequired(failed) else {
    fail("kernel close failure published readiness")
  }
  assert_eq(failed.cause(), CleanupRoot(role=KernelRoot, issue=Busy))
}

///|
test "model close and readiness mismatch both clean prepared worker" {
  let (model_events, model_result) = orchestration_fixture(fail_model=true)
  assert_eq(model_events, [
    "kernel-close", "worker-prepare", "model-close", "worker-cleanup",
  ])
  guard model_result is CleanupRequired(model_failed) else {
    fail("model close failure published readiness")
  }
  assert_eq(model_failed.cause(), CleanupRoot(role=ModelRoot, issue=Busy))
  let (mismatch_events, _) = orchestration_fixture(mismatch=true)
  assert_eq(mismatch_events, [
    "kernel-close", "worker-prepare", "model-close", "readiness", "worker-cleanup",
  ])
}

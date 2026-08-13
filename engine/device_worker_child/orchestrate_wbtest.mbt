///|
fn child_test_source() -> @worker_wire.EncodedBootstrapSource raise {
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
fn child_test_startup(
  source : @worker_wire.EncodedBootstrapSource,
  content? : String = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
) -> @worker_wire.WorkerStartupContract raise {
  @worker_wire.WorkerStartupContract::new(
    identity=@spec.ModelIdentity::new(
      @spec.ContentDigest::from_sha256(content),
      @spec.PlanDigest::from_sha256(
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      ),
    ),
    bootstrap_digest=@worker_wire.WorkerBootstrapDigest::from_sha256(
      "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
    ),
    bootstrap_source_digest=source.digest(),
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
      max_input_tokens=16,
      max_new_tokens=16,
      max_context_tokens=16,
      max_token_id=7,
      max_stop_token_ids=1,
      max_stop_strings=1,
      max_stop_string_bytes=8,
      max_trace_bytes=8,
      max_cache_scope_bytes=8,
      max_decoded_delta_bytes=8,
      max_deadline_millis=100UL,
      max_top_k=8,
      max_temperature=4.0,
    ),
    has_predecessor=false,
    predecessor_value=0UL,
  )
}

///|
fn child_phase_fixture(
  fail_configure? : Bool = false,
  fail_source? : Bool = false,
  fail_prepare? : Bool = false,
  cleanup_required? : Bool = false,
  mismatch? : Bool = false,
  fail_publish? : Bool = false,
  fail_eof? : Bool = false,
  fail_close? : Bool = false,
) -> (Array[String], Bool) raise {
  let source = child_test_source()
  let startup = child_test_startup(source)
  let mismatch_startup = child_test_startup(
    source,
    content="ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
  )
  let events : Array[String] = []
  let result = run_startup_with(
    () => {
      events.push("configure")
      if fail_configure {
        raise StartupFailed
      }
      startup
    },
    () => {
      events.push("source")
      if fail_source {
        raise StartupFailed
      }
      source
    },
    (_startup, _source) => {
      events.push("prepare")
      if fail_prepare {
        raise StartupFailed
      }
      if cleanup_required {
        BootstrapCleanup(9)
      } else {
        BootstrapReady(7)
      }
    },
    _failed => events.push("retry-cleanup"),
    _owner => {
      events.push("readiness")
      if mismatch {
        mismatch_startup
      } else {
        startup
      }
    },
    _actual => {
      events.push("ready")
      if fail_publish {
        raise StartupFailed
      }
    },
    () => {
      events.push("eof")
      if fail_eof {
        raise StartupFailed
      }
    },
    _owner => {
      events.push("close")
      if fail_close {
        raise StartupFailed
      }
    },
  )
  (events, result)
}

///|
test "startup phases consume Configure then source before real preparation" {
  let (events, result) = child_phase_fixture()
  assert_true(result)
  assert_eq(events, [
    "configure", "source", "prepare", "readiness", "ready", "eof", "close",
  ])
  let (configure_failure, configure_result) = child_phase_fixture(
    fail_configure=true,
  )
  assert_false(configure_result)
  assert_eq(configure_failure, ["configure"])
  let (source_failure, source_result) = child_phase_fixture(fail_source=true)
  assert_false(source_result)
  assert_eq(source_failure, ["configure", "source"])
}

///|
test "decode or bootstrap failure cannot publish Ready" {
  let (prepare_failure, prepare_result) = child_phase_fixture(fail_prepare=true)
  assert_false(prepare_result)
  assert_eq(prepare_failure, ["configure", "source", "prepare"])
  let (cleanup, cleanup_result) = child_phase_fixture(cleanup_required=true)
  assert_false(cleanup_result)
  assert_eq(cleanup, ["configure", "source", "prepare", "retry-cleanup"])
}

///|
test "exact readiness is queried before publication and mismatch closes owner" {
  let (events, result) = child_phase_fixture(mismatch=true)
  assert_false(result)
  assert_eq(events, ["configure", "source", "prepare", "readiness", "close"])
}

///|
test "Ready write and EOF failures still close the prepared owner" {
  let (write_events, write_result) = child_phase_fixture(fail_publish=true)
  assert_false(write_result)
  assert_eq(write_events, [
    "configure", "source", "prepare", "readiness", "ready", "close",
  ])
  let (eof_events, eof_result) = child_phase_fixture(fail_eof=true)
  assert_false(eof_result)
  assert_eq(eof_events, [
    "configure", "source", "prepare", "readiness", "ready", "eof", "close",
  ])
}

///|
test "owner close failure makes an otherwise clean child fail" {
  let (events, result) = child_phase_fixture(fail_close=true)
  assert_false(result)
  assert_eq(events, [
    "configure", "source", "prepare", "readiness", "ready", "eof", "close",
  ])
}

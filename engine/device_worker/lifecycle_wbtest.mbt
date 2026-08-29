///|
fn worker_test_startup() -> @worker_wire.WorkerStartupContract raise {
  let worker_limits = @worker_protocol.WorkerProtocolLimits::new(
    max_prefill_rows=1,
    max_decode_rows=1,
    max_plan_rows=1,
    max_plan_tokens=1,
    max_plan_pages=1,
    max_capabilities=1,
    max_completion_slots=1,
    max_sequence_tokens=1,
    max_token_id=0,
  )
  let inference_limits = @inference.InferenceLimits::new(
    max_text_bytes=1,
    max_input_tokens=1,
    max_new_tokens=1,
    max_context_tokens=1,
    max_token_id=0,
    max_stop_token_ids=1,
    max_stop_strings=1,
    max_stop_string_bytes=1,
    max_trace_bytes=1,
    max_cache_scope_bytes=1,
    max_decoded_delta_bytes=1,
    max_deadline_millis=1UL,
    max_top_k=1,
    max_temperature=1.0,
  )
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
    bootstrap_source_digest=@worker_wire.WorkerBootstrapSourceDigest::from_sha256(
      "112233445566778899aabbccddeeff00112233445566778899aabbccddeeff00",
    ),
    device_ordinal=0,
    model_generation_value=1UL,
    worker_limits~,
    inference_limits~,
    has_predecessor=false,
    predecessor_value=0UL,
  )
}

///|
fn empty_worker_resources() -> DeviceWorkerResources {
  {
    executor: None,
    weights: None,
    context: None,
    numeric_weights_owned_by_executor: false,
  }
}

///|
test "received startup must equal every independently admitted field" {
  let expected = worker_test_startup()
  let alternate_worker = @worker_protocol.WorkerProtocolLimits::new(
    max_prefill_rows=1,
    max_decode_rows=1,
    max_plan_rows=1,
    max_plan_tokens=1,
    max_plan_pages=1,
    max_capabilities=2,
    max_completion_slots=1,
    max_sequence_tokens=1,
    max_token_id=0,
  )
  let alternate_inference = @inference.InferenceLimits::new(
    max_text_bytes=2,
    max_input_tokens=1,
    max_new_tokens=1,
    max_context_tokens=1,
    max_token_id=0,
    max_stop_token_ids=1,
    max_stop_strings=1,
    max_stop_string_bytes=1,
    max_trace_bytes=1,
    max_cache_scope_bytes=1,
    max_decoded_delta_bytes=1,
    max_deadline_millis=1UL,
    max_top_k=1,
    max_temperature=1.0,
  )
  let variants = [
    @worker_wire.WorkerStartupContract::new(
      identity=expected.identity(),
      bootstrap_digest=expected.bootstrap_digest(),
      bootstrap_source_digest=@worker_wire.WorkerBootstrapSourceDigest::from_sha256(
        "212233445566778899aabbccddeeff00112233445566778899aabbccddeeff00",
      ),
      device_ordinal=expected.device_ordinal(),
      model_generation_value=expected.model_generation_value(),
      worker_limits=expected.worker_limits(),
      inference_limits=expected.inference_limits(),
      has_predecessor=false,
      predecessor_value=0UL,
    ),
    @worker_wire.WorkerStartupContract::new(
      identity=expected.identity(),
      bootstrap_digest=expected.bootstrap_digest(),
      bootstrap_source_digest=expected.bootstrap_source_digest(),
      device_ordinal=expected.device_ordinal(),
      model_generation_value=2UL,
      worker_limits=expected.worker_limits(),
      inference_limits=expected.inference_limits(),
      has_predecessor=false,
      predecessor_value=0UL,
    ),
    @worker_wire.WorkerStartupContract::new(
      identity=expected.identity(),
      bootstrap_digest=expected.bootstrap_digest(),
      bootstrap_source_digest=expected.bootstrap_source_digest(),
      device_ordinal=expected.device_ordinal(),
      model_generation_value=1UL,
      worker_limits=alternate_worker,
      inference_limits=expected.inference_limits(),
      has_predecessor=false,
      predecessor_value=0UL,
    ),
    @worker_wire.WorkerStartupContract::new(
      identity=expected.identity(),
      bootstrap_digest=expected.bootstrap_digest(),
      bootstrap_source_digest=expected.bootstrap_source_digest(),
      device_ordinal=expected.device_ordinal(),
      model_generation_value=1UL,
      worker_limits=expected.worker_limits(),
      inference_limits=alternate_inference,
      has_predecessor=false,
      predecessor_value=0UL,
    ),
    @worker_wire.WorkerStartupContract::new(
      identity=expected.identity(),
      bootstrap_digest=expected.bootstrap_digest(),
      bootstrap_source_digest=expected.bootstrap_source_digest(),
      device_ordinal=expected.device_ordinal(),
      model_generation_value=1UL,
      worker_limits=expected.worker_limits(),
      inference_limits=expected.inference_limits(),
      has_predecessor=true,
      predecessor_value=1UL,
    ),
  ]
  for received in variants {
    try validate_received_startup(received, expected) catch {
      Invalid(StartupContract) => ()
      _ => fail("unexpected received-startup error")
    } noraise {
      _ => fail("foreign received startup was accepted")
    }
  }
  validate_received_startup(expected, expected)
}

///|
test "readiness rejects an incomplete aggregate and close is idempotent" {
  let startup = worker_test_startup()
  let owner : DeviceWorkerOwner = {
    startup,
    resources: empty_worker_resources(),
    lifecycle: WorkerReady,
  }
  try owner.readiness_contract() catch {
    Invalid(ResourceState) => ()
    _ => fail("unexpected incomplete readiness error")
  } noraise {
    _ => fail("empty aggregate published readiness")
  }
  owner.close()
  assert_eq(owner.lifecycle(), WorkerClosed)
  owner.close()
  try owner.readiness_contract() catch {
    InvalidLifecycle(WorkerClosed) => ()
    _ => fail("unexpected closed readiness error")
  } noraise {
    _ => fail("closed aggregate remained ready")
  }
}

///|
test "failed preparation cleanup remains retry-safe and idempotent" {
  let failed : FailedDeviceWorkerPreparation = {
    resources: empty_worker_resources(),
    cause: Invalid(StartupContract),
    cleanup_failure: Invalid(StartupContract),
    lifecycle: WorkerClosingFailed,
  }
  assert_eq(failed.cause(), Invalid(StartupContract))
  assert_eq(failed.cleanup_failure(), Invalid(StartupContract))
  assert_false(failed.is_closed())
  failed.retry_close()
  assert_true(failed.is_closed())
  failed.retry_close()
  assert_true(failed.is_closed())
}

///|
test "cleanup failure blocks parents and a retry preserves exact order" {
  let calls : Array[String] = []
  let attempts = Ref(0)
  try
    run_cleanup_chain(
      () => {
        calls.push("executor")
        attempts.val += 1
        if attempts.val == 1 {
          raise Invalid(StartupContract)
        }
      },
      () => calls.push("weights"),
      () => calls.push("context"),
    )
  catch {
    Invalid(StartupContract) => ()
    _ => fail("unexpected child cleanup error")
  } noraise {
    _ => fail("first child cleanup unexpectedly succeeded")
  }
  assert_eq(calls, ["executor"])
  run_cleanup_chain(() => calls.push("executor"), () => calls.push("weights"), () => {
    calls.push("context")
  })
  assert_eq(calls, ["executor", "executor", "weights", "context"])
}

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
  { executor: None, weights: None, context: None }
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
test "model metadata canonically binds the paged plan identity" {
  let metadata = @spec.LlamaModelMetadata::from_verified_content(
    @spec.LlamaModelSpec::new(
      vocabulary_size=8,
      hidden_size=4,
      intermediate_size=8,
      layer_count=1,
      attention_head_count=2,
      kv_head_count=1,
      context_length=8,
      rope_theta=10000.0,
      weight_dtype=BFloat16,
      kv_dtype=BFloat16,
    ),
    @spec.ContentDigest::from_sha256(
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
  )
  let paged = @llama.build_paged(metadata)
  assert_true(metadata.identity().plan() != paged.identity().plan())
  validate_model_identity(metadata, paged.identity())
  try validate_model_identity(metadata, metadata.identity()) catch {
    Invalid(ModelIdentity) => ()
    _ => fail("unexpected arbitrary plan-digest error")
  } noraise {
    _ => fail("same-content arbitrary plan digest was accepted")
  }
  let different_spec = @spec.LlamaModelMetadata::from_verified_content(
    @spec.LlamaModelSpec::new(
      vocabulary_size=8,
      hidden_size=4,
      intermediate_size=8,
      layer_count=1,
      attention_head_count=2,
      kv_head_count=1,
      context_length=8,
      rope_theta=20000.0,
      rms_norm_epsilon=0.00001,
      weight_dtype=BFloat16,
      kv_dtype=BFloat16,
    ),
    metadata.identity().content(),
  )
  try validate_model_identity(different_spec, paged.identity()) catch {
    Invalid(ModelIdentity) => ()
    _ => fail("unexpected foreign semantic-plan error")
  } noraise {
    _ => fail("different RoPE/epsilon metadata was accepted")
  }
  let foreign = @spec.ModelIdentity::new(
    @spec.ContentDigest::from_sha256(
      "1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
    paged.identity().plan(),
  )
  try validate_model_identity(metadata, foreign) catch {
    Invalid(ModelIdentity) => ()
    _ => fail("unexpected foreign model-content error")
  } noraise {
    _ => fail("foreign model content was accepted")
  }
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

///|
fn expect_pinned_release_rejection(
  recipe : String,
  content : String,
  plan : String,
  tokenizer : String,
  worker : String,
  expected_worker : String,
) -> Unit {
  try
    require_pinned_release_values(
      recipe, content, plan, tokenizer, worker, expected_worker,
    )
  catch {
    PinnedReleaseMismatch => ()
    _ => abort("wrong pinned-release hostile result")
  } noraise {
    _ => abort("hostile release identity was admitted")
  }
}

///|
test "pinned tiny model identity is the upstream BF16 single row graph" {
  let identity = pinned_tiny_model_identity()
  assert_eq(identity.content().as_hex(), PINNED_MODEL_CONTENT_SHA256)
  inspect(
    identity.plan().as_hex(),
    content="22103eddad1277154b79321e53893ad579e47d040d62b896f5596624e0b7be3e",
  )
  require_pinned_release_values(
    PINNED_RUNTIME_RECIPE,
    identity.content().as_hex(),
    identity.plan().as_hex(),
    PINNED_TOKENIZER_SHA256,
    "a".repeat(64),
    "a".repeat(64),
  )
}

///|
test "pinned campaign rejects recipe model tokenizer and child substitution" {
  let identity = pinned_tiny_model_identity()
  let content = identity.content().as_hex()
  let plan = identity.plan().as_hex()
  let worker = "a".repeat(64)
  expect_pinned_release_rejection(
    "dense_llama_i8_paged_aot_v6",
    content,
    plan,
    PINNED_TOKENIZER_SHA256,
    worker,
    worker,
  )
  expect_pinned_release_rejection(
    PINNED_RUNTIME_RECIPE,
    "b".repeat(64),
    plan,
    PINNED_TOKENIZER_SHA256,
    worker,
    worker,
  )
  expect_pinned_release_rejection(
    PINNED_RUNTIME_RECIPE,
    content,
    "c".repeat(64),
    PINNED_TOKENIZER_SHA256,
    worker,
    worker,
  )
  expect_pinned_release_rejection(
    PINNED_RUNTIME_RECIPE,
    content,
    plan,
    "d".repeat(64),
    worker,
    worker,
  )
  expect_pinned_release_rejection(
    PINNED_RUNTIME_RECIPE,
    content,
    plan,
    PINNED_TOKENIZER_SHA256,
    worker,
    "e".repeat(64),
  )
}

///|
test "campaign request pins the two-token production framed contract" {
  let limits = campaign_inference_limits()
  let frame = campaign_request_frame()
  let request = @framed_wire.RequestFrameBuffer::new(
      @framed_wire.FramedWireLimits::new(limits, max_frame_bytes=1048576),
    )
    .load(frame, frame.length())
    .request()
  assert_eq(request.model_identity(), pinned_tiny_model_identity())
  guard request.input() is TokenIds(input) else {
    abort("campaign request did not preserve token input")
  }
  assert_eq(input.length(), 1)
  assert_eq(input.token_at(0), 1)
  assert_eq(request.max_new_tokens(), 2)
  assert_eq(request.context_ceiling(), 256)
  assert_eq(campaign_inference_limits().max_context_tokens(), 256)
  assert_eq(CAMPAIGN_TOKENS_PER_PAGE, 8)
  assert_eq(CAMPAIGN_CONTEXT_PAGES, 32)
  assert_eq(
    CAMPAIGN_CONTEXT_TOKENS / CAMPAIGN_TOKENS_PER_PAGE,
    CAMPAIGN_CONTEXT_PAGES,
  )
  assert_eq(request.sampling().mode(), Greedy)
  assert_eq(request.cache().permission(), Disabled)
}

///|
test "expected-token substitutions do not alter the encoded request" {
  let canonical_frame = campaign_request_frame()
  ignore(campaign_validation_contract())
  ignore(campaign_validation_contract(first_token=2049))
  ignore(campaign_validation_contract(second_token=688))
  assert_eq(campaign_request_frame(), canonical_frame)
}

///|
fn expect_serving_contract_rejection(
  frame : FixedArray[Byte],
  first_token? : Int = 1031,
  tokens_per_page? : Int = CAMPAIGN_TOKENS_PER_PAGE,
) -> Unit {
  try
    @runtime_instance.RuntimeSpawnedServingValidationContract::new(
      frame,
      pinned_tiny_model_identity(),
      first_token,
      2185,
      1,
      CAMPAIGN_CONTEXT_TOKENS,
      tokens_per_page,
      campaign_inference_limits(),
      1048576,
      1048576,
      5000,
    )
  catch {
    @runtime_instance.RuntimeSpawnedServingValidationFailed(
      SpawnedServingContract
    ) => ()
    _ => abort("serving contract failed at the wrong boundary")
  } noraise {
    _ => abort("hostile serving contract was admitted")
  }
}

///|
test "serving campaign pins and authenticates the exact bounded frame" {
  ignore(campaign_serving_validation_contract())
  ignore(campaign_serving_validation_contract(first_token=2049))
  ignore(campaign_serving_validation_contract(second_token=688))
  let corrupt = campaign_request_frame()
  corrupt[0] = corrupt[0] ^ b'\x01'
  expect_serving_contract_rejection(corrupt)
  expect_serving_contract_rejection(campaign_request_frame(), first_token=3000)
  expect_serving_contract_rejection(campaign_request_frame(), tokens_per_page=2)
}

///|
test "serving evidence scope excludes TLS and performance claims" {
  let scope = "traffic_readiness_observed=1\n" +
    "native_tcp_listener=pass\n" +
    "event_order=accepted,token,token,usage,completed\n" +
    "tls_validation=not-run\nperformance_validation=not-run\n"
  assert_true(scope.contains("traffic_readiness_observed=1"))
  assert_true(scope.contains("tls_validation=not-run"))
  assert_true(scope.contains("performance_validation=not-run"))
  assert_false(scope.contains("tls_validation=pass"))
  assert_false(scope.contains("performance_validation=pass"))
}

///|
test "broad BF16 campaign is qualification-only and names unexercised cases" {
  ignore(broad_campaign_validation_contract())
  let scope = "campaign_scope=qualification-only\n" +
    "production_readiness=not-claimed\n" +
    "performance_baseline=not-established\n" +
    "mixed_concurrent_requests=pass\n" +
    "cancellation_isolation=pass\n" +
    "malformed_native_frame=fail-closed-connection-isolation-pass\n" +
    "post_malformed_same_owner_recovery=pass\n" +
    "malformed_http=not-exercised-native-framed-owner\n" +
    "overload_rejection=not-exercised-single-connection-owner\n"
  assert_true(scope.contains("campaign_scope=qualification-only"))
  assert_true(scope.contains("fail-closed-connection-isolation-pass"))
  assert_true(scope.contains("post_malformed_same_owner_recovery=pass"))
  assert_true(scope.contains("malformed_http=not-exercised"))
  assert_false(scope.contains("production_readiness=pass"))
  assert_false(scope.contains("performance_baseline=pass"))
}

///|
test "campaign evidence is execution-only and explicitly not selected logits" {
  let expected = "spawned_child_executor_prepared=1\n" +
    "traffic_readiness=0\nplan_executed=2\nprefill_plan_executed=1\n" +
    "same_page_decode_plan_executed=1\ngenerated_tokens=1031,2185\n" +
    "context_capacity_tokens=256\n" +
    "emergency_decode_page_reserve=positive-admitted\n" +
    "token_numerical_correctness=pinned-corpus-pass\n" +
    "selected_logit_correctness=not-observed\n" +
    "cleanup_complete=1\nspawned_execution=pass\n"
  assert_true(expected.contains("traffic_readiness=0"))
  assert_true(expected.contains("plan_executed=2"))
  assert_false(expected.contains("traffic_readiness=1"))
  assert_false(expected.contains("selected_logit_correctness=pass"))
}

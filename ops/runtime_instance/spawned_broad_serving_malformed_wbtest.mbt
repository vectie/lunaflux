///|
fn broad_serving_malformed_metric_fixture(
  admissions? : UInt64 = 4UL,
  completions? : UInt64 = 3UL,
  prompt_tokens? : UInt64 = 4UL,
  generated_tokens? : UInt64 = 5UL,
  accepts? : UInt64 = 4UL,
  disconnects? : UInt64 = 4UL,
  rejections? : UInt64 = 2UL,
  queue_depth? : Int = 0,
  kv_pages_free? : Int = 32,
  backpressure? : UInt64 = 1UL,
) -> BroadServingMetricSample {
  {
    admissions,
    completions,
    cancellations: 1UL,
    deadlines: 0UL,
    failures: 0UL,
    prompt_tokens,
    generated_tokens,
    worker_failures: 0UL,
    backpressure,
    network_accepts: accepts,
    network_disconnects: disconnects,
    network_rejections: rejections,
    queue_depth,
    active_requests: 0,
    kv_pages_used: 0,
    kv_pages_free,
  }
}

///|
fn broad_serving_final_metric_fixture(
  admissions? : UInt64 = 5UL,
  completions? : UInt64 = 4UL,
  prompt_tokens? : UInt64 = 5UL,
  generated_tokens? : UInt64 = 7UL,
  accepts? : UInt64 = 5UL,
  disconnects? : UInt64 = 5UL,
  rejections? : UInt64 = 2UL,
  queue_depth? : Int = 0,
  kv_pages_free? : Int = 32,
  backpressure? : UInt64 = 1UL,
) -> BroadServingMetricSample {
  broad_serving_malformed_metric_fixture(
    admissions~,
    completions~,
    prompt_tokens~,
    generated_tokens~,
    accepts~,
    disconnects~,
    rejections~,
    queue_depth~,
    kv_pages_free~,
    backpressure~,
  )
}

///|
test "broad serving malformed retirement and reuse balances are exact" {
  let malformed = broad_serving_malformed_metric_fixture()
  assert_true(broad_serving_malformed_request_metrics_valid(malformed, 1))
  assert_true(broad_serving_malformed_network_metrics_valid(malformed))
  assert_true(broad_serving_resource_metrics_valid(malformed, 32))
  assert_false(
    broad_serving_malformed_request_metrics_valid(
      broad_serving_malformed_metric_fixture(backpressure=0UL),
      1,
    ),
  )
  assert_false(
    broad_serving_malformed_network_metrics_valid(
      broad_serving_malformed_metric_fixture(rejections=1UL),
    ),
  )
  assert_false(
    broad_serving_resource_metrics_valid(
      broad_serving_malformed_metric_fixture(queue_depth=1),
      32,
    ),
  )

  let final_sample = broad_serving_final_metric_fixture()
  assert_true(broad_serving_final_request_metrics_valid(final_sample, 1))
  assert_true(broad_serving_final_network_metrics_valid(final_sample))
  assert_true(broad_serving_resource_metrics_valid(final_sample, 32))
  assert_false(
    broad_serving_final_request_metrics_valid(
      broad_serving_final_metric_fixture(generated_tokens=6UL),
      1,
    ),
  )
  assert_false(
    broad_serving_final_network_metrics_valid(
      broad_serving_final_metric_fixture(disconnects=4UL),
    ),
  )
  assert_false(
    broad_serving_resource_metrics_valid(
      broad_serving_final_metric_fixture(kv_pages_free=31),
      32,
    ),
  )
}

///|
test "broad serving malformed diagnostics are stable and payload free" {
  let rules : Array[(RuntimeSpawnedBroadServingValidationRule, String)] = [
    (BroadServingMalformedTransition, "malformed_transition"),
    (
      BroadServingMalformedUnexpectedOwnerFailure,
      "malformed_unexpected_owner_failure",
    ),
    (BroadServingMalformedRequestMetrics, "malformed_metrics_request"),
    (BroadServingMalformedNetworkMetrics, "malformed_metrics_network"),
    (BroadServingMalformedResourceMetrics, "malformed_metrics_resources"),
    (BroadServingMalformedReuseAccepted, "malformed_reuse_events_accepted"),
    (BroadServingMalformedReuseToken0, "malformed_reuse_events_token_0"),
    (BroadServingMalformedReuseToken1, "malformed_reuse_events_token_1"),
    (BroadServingMalformedReuseUsage, "malformed_reuse_events_usage"),
    (BroadServingMalformedReuseCompleted, "malformed_reuse_events_completed"),
    (
      BroadServingMalformedReuseRequestMetrics,
      "malformed_reuse_metrics_request",
    ),
    (
      BroadServingMalformedReuseNetworkMetrics,
      "malformed_reuse_metrics_network",
    ),
    (
      BroadServingMalformedReuseResourceMetrics,
      "malformed_reuse_metrics_resources",
    ),
  ]
  for entry in rules {
    let (rule, expected) = entry
    assert_eq(rule.code(), expected)
  }
}

///|
test "broad serving malformed metric failures retain their category" {
  let contract = broad_serving_wb_contract(
    broad_serving_wb_frame(1UL, 1, false),
  )
  require_broad_serving_malformed_metrics(
    broad_serving_malformed_metric_fixture(),
    contract,
    32,
  )
  require_broad_serving_final_metrics(
    broad_serving_final_metric_fixture(),
    contract,
    32,
  )
  let cases : Array[
    (BroadServingMetricSample, RuntimeSpawnedBroadServingValidationRule),
  ] = [
    (
      broad_serving_malformed_metric_fixture(admissions=3UL),
      BroadServingMalformedRequestMetrics,
    ),
    (
      broad_serving_malformed_metric_fixture(rejections=1UL),
      BroadServingMalformedNetworkMetrics,
    ),
    (
      broad_serving_malformed_metric_fixture(kv_pages_free=31),
      BroadServingMalformedResourceMetrics,
    ),
  ]
  for entry in cases {
    let (sample, expected) = entry
    assert_true(
      broad_serving_wb_expect_rule(
        () => require_broad_serving_malformed_metrics(sample, contract, 32),
        expected,
      ),
    )
  }
  let reuse_cases : Array[
    (BroadServingMetricSample, RuntimeSpawnedBroadServingValidationRule),
  ] = [
    (
      broad_serving_final_metric_fixture(completions=3UL),
      BroadServingMalformedReuseRequestMetrics,
    ),
    (
      broad_serving_final_metric_fixture(accepts=4UL),
      BroadServingMalformedReuseNetworkMetrics,
    ),
    (
      broad_serving_final_metric_fixture(queue_depth=1),
      BroadServingMalformedReuseResourceMetrics,
    ),
  ]
  for entry in reuse_cases {
    let (sample, expected) = entry
    assert_true(
      broad_serving_wb_expect_rule(
        () => require_broad_serving_final_metrics(sample, contract, 32),
        expected,
      ),
    )
  }
}

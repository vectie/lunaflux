///|
test "control health and readiness projections are exact across lifecycle" {
  assert_true(luna_online_control_health_available(0, false))
  assert_false(luna_online_control_health_available(1, false))
  assert_false(luna_online_control_health_available(0, true))

  assert_true(
    luna_online_control_readiness_available(0, false, false, 1, false, true),
  )
  assert_false(
    luna_online_control_readiness_available(1, false, false, 1, false, true),
  )
  assert_false(
    luna_online_control_readiness_available(0, true, false, 1, false, true),
  )
  assert_false(
    luna_online_control_readiness_available(0, false, true, 1, false, true),
  )
  assert_false(
    luna_online_control_readiness_available(0, false, false, 0, false, true),
  )
  assert_false(
    luna_online_control_readiness_available(0, false, false, 1, true, true),
  )
  assert_false(
    luna_online_control_readiness_available(0, false, false, 1, false, false),
  )
}

///|
test "control socket polling is independently strict-bounded" {
  assert_eq(luna_online_http_control_poll_millis(1), 1)
  assert_eq(luna_online_http_control_poll_millis(25), 1)
  assert_eq(luna_online_http_control_poll_millis(5000), 1)
}

///|
test "startup measurement publication lifecycle and bounds are exact" {
  assert_true(luna_online_startup_measurement_valid(0))
  assert_true(luna_online_startup_measurement_valid(86400000))
  assert_false(luna_online_startup_measurement_valid(-1))
  assert_false(luna_online_startup_measurement_valid(86400001))
  assert_true(
    luna_online_startup_measurement_publish_allowed(false, false, false, 0),
  )
  for
    tuple in [
      (true, false, false, 0),
      (false, true, false, 0),
      (false, false, true, 0),
      (false, false, false, 1),
    ] {
    let (published, failed, draining, phase) = tuple
    assert_false(
      luna_online_startup_measurement_publish_allowed(
        published, failed, draining, phase,
      ),
    )
  }
}

///|
fn render_control_metrics_test(
  metrics : @metrics.LunaInstanceMetrics,
  scratch : LunaOnlineTcpOutputScratch,
  cold_start_latency_millis : Int,
  readiness : Bool,
) -> String {
  let snapshot = try! metrics.snapshot()
  let write = try! scratch.begin_write()
  let length = try! luna_online_render_control_metrics_response(
    write, snapshot, cold_start_latency_millis, readiness,
  )
  let flight = try! write.publish(length)
  let encoded = Bytes::makei(length, index => try! flight.byte_at(index))
  try! flight.release()
  try! @utf8.decode(encoded)
}

///|
test "live control metrics project changing counters gauges and readiness" {
  let metrics = @metrics.LunaInstanceMetrics::new()
  let scratch = try! LunaOnlineTcpOutputScratch::new(
    LUNA_ONLINE_CONTROL_METRICS_RESPONSE_CAPACITY,
  )
  try! metrics.record_counter(Admissions, amount=2)
  try! metrics.record_counter(GeneratedTokens, amount=7)
  try! metrics.set_gauge(QueueDepth, 3)
  try! metrics.set_gauge(KvPagesFree, 29)
  let first = render_control_metrics_test(metrics, scratch, 41, true)
  assert_true(first.has_prefix("HTTP/1.1 200 OK\r\n"))
  assert_true(first.contains("\"schema\":\"lunaflux.runtime.metrics.v1\""))
  assert_true(first.contains("\"cold_start_latency_millis\":41"))
  assert_true(first.contains("\"readiness\":true"))
  assert_true(first.contains("\"admissions\":2"))
  assert_true(first.contains("\"generated_tokens\":7"))
  assert_true(first.contains("\"queue_depth\":3"))
  assert_true(first.contains("\"kv_pages_free\":29"))

  try! metrics.record_counter(Admissions, amount=5)
  try! metrics.set_gauge(QueueDepth, 0)
  let second = render_control_metrics_test(metrics, scratch, 41, false)
  assert_true(second.has_prefix("HTTP/1.1 503 Service Unavailable\r\n"))
  assert_true(second.contains("\"readiness\":false"))
  assert_true(second.contains("\"admissions\":7"))
  assert_true(second.contains("\"queue_depth\":0"))
  assert_false(second.contains("\"admissions\":2"))
}

///|
test "live control metrics include the exact bounded counter and gauge vocabulary" {
  let metrics = @metrics.LunaInstanceMetrics::new()
  let scratch = try! LunaOnlineTcpOutputScratch::new(
    LUNA_ONLINE_CONTROL_METRICS_RESPONSE_CAPACITY,
  )
  let text = render_control_metrics_test(metrics, scratch, -1, true)
  assert_true(text.contains("\"cold_start_latency_millis\":null"))
  assert_true(text.contains("\"readiness\":false"))
  for
    name in [
      "admissions", "completions", "cancellations", "deadlines", "failures", "prompt_tokens",
      "generated_tokens", "worker_restarts", "worker_failures", "network_accepts",
      "network_disconnects", "network_rejections", "backpressure", "prefix_lookups",
      "prefix_hits", "prefix_misses", "prefix_evictions", "prefix_tokens_reused",
      "prefix_tokens_computed", "prefix_publications", "graph_hits", "graph_misses",
      "queue_depth", "active_requests", "kv_pages_used", "kv_pages_free", "prefix_entries",
      "prefix_pages",
    ] {
    assert_true(text.contains("\"" + name + "\":"))
  }
  assert_true(text.length() < LUNA_ONLINE_CONTROL_METRICS_RESPONSE_CAPACITY)
}

///|
test "live control metrics maximum scalar projection remains within startup backing" {
  let metrics = @metrics.LunaInstanceMetrics::new()
  let counters : FixedArray[@metrics.LunaInstanceCounter] = [
    Admissions,
    Completions,
    Cancellations,
    Deadlines,
    Failures,
    PromptTokens,
    GeneratedTokens,
    WorkerRestarts,
    WorkerFailures,
    NetworkAccepts,
    NetworkDisconnects,
    NetworkRejections,
    Backpressure,
    PrefixLookups,
    PrefixHits,
    PrefixMisses,
    PrefixEvictions,
    PrefixTokensReused,
    PrefixTokensComputed,
    PrefixPublications,
    GraphHits,
    GraphMisses,
  ]
  for counter in counters {
    metrics.record_counter_u64(counter, 0xffffffffffffffffUL)
  }
  let gauges : FixedArray[@metrics.LunaInstanceGauge] = [
    QueueDepth,
    ActiveRequests,
    KvPagesUsed,
    KvPagesFree,
    PrefixEntries,
    PrefixPages,
  ]
  for gauge in gauges {
    try! metrics.set_gauge(gauge, 0x7fffffff)
  }
  let scratch = try! LunaOnlineTcpOutputScratch::new(
    LUNA_ONLINE_CONTROL_METRICS_RESPONSE_CAPACITY,
  )
  let text = render_control_metrics_test(metrics, scratch, 86400000, true)
  assert_true(text.contains("18446744073709551615"))
  assert_true(text.contains("2147483647"))
  assert_true(text.length() < LUNA_ONLINE_CONTROL_METRICS_RESPONSE_CAPACITY)
}

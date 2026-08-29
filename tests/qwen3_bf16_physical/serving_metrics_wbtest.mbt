///|
fn qwen3_metrics_fixture(
  admissions? : Int = 1,
  kv_pages_used? : Int = 0,
  readiness? : String = "true",
) -> Bytes {
  @utf8.encode(
    "{\"schema\":\"lunaflux.runtime.metrics.v1\",\"cold_start_latency_millis\":41,\"readiness\":" +
    readiness +
    ",\"counters\":{\"admissions\":" +
    admissions.to_string() +
    ",\"completions\":1,\"cancellations\":0,\"deadlines\":0,\"failures\":0,\"prompt_tokens\":2,\"generated_tokens\":3,\"worker_restarts\":0,\"worker_failures\":0,\"network_accepts\":1,\"network_disconnects\":1,\"network_rejections\":0,\"backpressure\":0,\"prefix_lookups\":0,\"prefix_hits\":0,\"prefix_misses\":0,\"prefix_evictions\":0,\"prefix_tokens_reused\":0,\"prefix_tokens_computed\":0,\"prefix_publications\":0,\"graph_hits\":0,\"graph_misses\":1},\"gauges\":{\"queue_depth\":0,\"active_requests\":0,\"kv_pages_used\":" +
    kv_pages_used.to_string() +
    ",\"kv_pages_free\":32,\"prefix_entries\":0,\"prefix_pages\":0}}\n",
  )
}

///|
test "Qwen serving metrics require terminal request network and KV balance" {
  qwen3_verify_serving_metrics(qwen3_metrics_fixture(), 2, 3, 32)
}

///|
test "Qwen serving metrics reject readiness request and KV drift" {
  for
    hostile in [
      qwen3_metrics_fixture(admissions=2),
      qwen3_metrics_fixture(kv_pages_used=1),
      qwen3_metrics_fixture(readiness="false"),
    ] {
    try qwen3_verify_serving_metrics(hostile, 2, 3, 32) catch {
      InvalidInput(_) | CorpusMismatch(_) => ()
      _ => fail("unexpected hostile metrics error")
    } noraise {
      _ => fail("hostile Qwen serving metrics were admitted")
    }
  }
}

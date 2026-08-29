///|
fn qwen3_metrics_exact_int(
  value : Json,
  label : String,
) -> Int raise Qwen3PhysicalError {
  guard value is Number(number, repr=_) else {
    raise CorpusMismatch(label + " is not numeric")
  }
  let integer = number.to_int()
  qwen3_require(integer.to_double() == number, label + " is not exact")
  integer
}

///|
fn qwen3_metrics_field(
  fields : Map[String, Json],
  name : String,
) -> Int raise Qwen3PhysicalError {
  guard fields.get(name) is Some(value) else {
    raise CorpusMismatch("metrics field is absent: " + name)
  }
  qwen3_metrics_exact_int(value, "metrics field " + name)
}

///|
fn qwen3_verify_metrics_fields(
  counters : Map[String, Json],
  gauges : Map[String, Json],
  input_tokens : Int,
  output_tokens : Int,
  kv_pages_free : Int,
) -> Unit raise Qwen3PhysicalError {
  qwen3_require(counters.length() == 22, "metrics counter vocabulary differs")
  qwen3_require(gauges.length() == 6, "metrics gauge vocabulary differs")
  for
    name in [
      "cancellations", "deadlines", "failures", "worker_restarts", "worker_failures",
      "network_rejections", "backpressure", "prefix_lookups", "prefix_hits", "prefix_misses",
      "prefix_evictions", "prefix_tokens_reused", "prefix_tokens_computed", "prefix_publications",
    ] {
    qwen3_require(qwen3_metrics_field(counters, name) == 0, name + " differs")
  }
  qwen3_require(
    qwen3_metrics_field(counters, "admissions") == 1,
    "admissions differ",
  )
  qwen3_require(
    qwen3_metrics_field(counters, "completions") == 1,
    "completions differ",
  )
  qwen3_require(
    qwen3_metrics_field(counters, "prompt_tokens") == input_tokens,
    "prompt tokens differ",
  )
  qwen3_require(
    qwen3_metrics_field(counters, "generated_tokens") == output_tokens,
    "generated tokens differ",
  )
  qwen3_require(
    qwen3_metrics_field(counters, "network_accepts") == 1,
    "network accepts differ",
  )
  qwen3_require(
    qwen3_metrics_field(counters, "network_disconnects") == 1,
    "network disconnects differ",
  )
  for name in ["graph_hits", "graph_misses"] {
    qwen3_require(
      qwen3_metrics_field(counters, name) >= 0,
      name + " is negative",
    )
  }
  for
    name in [
      "queue_depth", "active_requests", "kv_pages_used", "prefix_entries", "prefix_pages",
    ] {
    qwen3_require(qwen3_metrics_field(gauges, name) == 0, name + " differs")
  }
  qwen3_require(
    qwen3_metrics_field(gauges, "kv_pages_free") == kv_pages_free,
    "free KV pages differ",
  )
}

///|
fn qwen3_verify_serving_metrics(
  bytes : BytesView,
  input_tokens : Int,
  output_tokens : Int,
  kv_pages_free : Int,
) -> Unit raise Qwen3PhysicalError {
  let parsed = @json.parse(@utf8.decode(bytes)) catch {
    _ => raise CorpusMismatch("serving metrics JSON is malformed")
  }
  guard parsed is Object(root) &&
    root.length() == 5 &&
    root.get("schema") is Some(String("lunaflux.runtime.metrics.v1")) &&
    root.get("readiness") is Some(True) &&
    root.get("cold_start_latency_millis") is Some(cold_start) &&
    root.get("counters") is Some(Object(counters)) &&
    root.get("gauges") is Some(Object(gauges)) else {
    raise CorpusMismatch("serving metrics schema or readiness differs")
  }
  let cold_start_millis = qwen3_metrics_exact_int(
    cold_start, "cold-start latency",
  )
  qwen3_require(
    cold_start_millis >= 0 && cold_start_millis <= 86400000,
    "cold-start latency is outside bounds",
  )
  qwen3_verify_metrics_fields(
    counters, gauges, input_tokens, output_tokens, kv_pages_free,
  )
}

///|
async fn qwen3_metrics_check(
  args : ArrayView[String],
) -> Unit raise Qwen3PhysicalError {
  qwen3_require(args.length() == 7, "metrics-check arguments are invalid")
  let input_tokens = qwen3_serving_int(args[4], "metrics input tokens")
  let output_tokens = qwen3_serving_int(args[5], "metrics output tokens")
  let kv_pages_free = qwen3_serving_int(args[6], "metrics free KV pages")
  let bytes = qwen3_read_pinned(args[2], args[3], 1048576)
  qwen3_verify_serving_metrics(
    bytes, input_tokens, output_tokens, kv_pages_free,
  )
  println(
    "outcome=qwen3-serving-metrics-pass queue_depth=0 active_requests=0 kv_pages_used=0 kv_pages_free=" +
    kv_pages_free.to_string() +
    " network_accepts=1 network_disconnects=1 worker_failures=0",
  )
}

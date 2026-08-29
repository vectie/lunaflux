///|
fn local_benchmark_test_release() -> RuntimeReleaseAdmission {
  {
    runtime_recipe: "dense_llama_paged_aot_v5",
    model_content_sha256: "a".repeat(64),
    model_plan_sha256: "b".repeat(64),
    tokenizer_sha256: "c".repeat(64),
    worker_executable_sha256: "d".repeat(64),
    sampling_runtime: HostSamplingRuntime,
    has_fused_runtime: false,
    evidence: "release evidence",
    evidence_sha256: "e".repeat(64),
  }
}

///|
fn local_benchmark_test_measurements() -> @benchmark_evidence.BenchmarkSummaryEvidence raise {
  let event = @benchmark_evidence.BenchmarkRequestEvent::new(
    request_ordinal=0,
    outcome=Completed,
    submitted_at_ns=1000L,
    admitted_at_ns=Some(1100L),
    first_token_at_ns=Some(1200L),
    terminal_at_ns=1500L,
    input_tokens=1L,
    generated_tokens=2L,
  )
  @benchmark_evidence.BenchmarkSummaryEvidence::admit(
    engine=LunaFlux,
    profile=Latency,
    ordinal=1,
    order_position=0,
    declared_submitted=1,
    measurement=@benchmark_runner.system_monotonic_millis_measurement_digest(),
    events=[event],
  )
}

///|
fn local_benchmark_test_result(
  readiness? : RuntimeInstanceReadiness = RuntimeReady,
) -> RuntimeSpawnedBenchmarkResult raise {
  {
    measurements: local_benchmark_test_measurements(),
    startup: { cold_start_latency_millis: 123, readiness },
    serving: {
      first_token: 17,
      second_token: 23,
      event_count: 5,
      kv_pages_free: 31,
      network_accepts: 1UL,
      network_disconnects: 1UL,
    },
  }
}

///|
test "local benchmark evidence retains raw timings and no promotion authority" {
  let evidence = local_benchmark_evidence(
    local_benchmark_test_release(),
    local_benchmark_test_result(),
  )
  assert_eq(evidence.cold_start_latency_millis(), 123)
  assert_eq(evidence.startup_readiness(), RuntimeReady)
  assert_eq(evidence.first_token(), 17)
  assert_eq(evidence.second_token(), 23)
  assert_false(evidence.raw_events_canonical_bytes().is_empty())
  assert_false(evidence.summary_canonical_bytes().is_empty())
  assert_false(evidence.promotion_authorized())
  assert_false(evidence.comparison_admitted())
  let canonical = @utf8.decode(evidence.canonical_bytes())
  assert_true(
    canonical.contains("schema=lunaflux-local-runtime-benchmark.v1\n"),
  )
  assert_true(canonical.contains("cold_start_latency_millis=123\n"))
  assert_true(canonical.contains("startup_readiness=true\n"))
  assert_true(canonical.contains("raw_events_canonical_hex="))
  assert_true(canonical.contains("summary_canonical_hex="))
  assert_true(canonical.contains("terminal_phase=closed\n"))
  assert_true(canonical.contains("terminal_readiness=false\n"))
  assert_true(canonical.contains("comparison_admission=not-run\n"))
  assert_true(canonical.contains("promotion_authority=absent\n"))
}

///|
test "local benchmark evidence rejects false startup readiness" {
  try
    local_benchmark_evidence(
      local_benchmark_test_release(),
      local_benchmark_test_result(readiness=RuntimeNotReady),
    )
  catch {
    RuntimeLocalBenchmarkFailed(LocalBenchmarkEvidence) => ()
    _ => abort("unexpected false-readiness benchmark rejection")
  } noraise {
    _ => abort("false-readiness benchmark evidence was admitted")
  }
}

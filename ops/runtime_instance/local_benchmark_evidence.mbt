///|
const LOCAL_BENCHMARK_MAXIMUM_EVIDENCE_BYTES : Int = 65536

///|
fn local_benchmark_evidence(
  release : RuntimeReleaseAdmission,
  result : RuntimeSpawnedBenchmarkResult,
) -> RuntimeLocalBenchmarkEvidence raise RuntimeLocalBenchmarkError {
  let startup = result.startup_measurement()
  if startup.readiness() != RuntimeReady ||
    startup.cold_start_latency_millis() < 0 ||
    startup.cold_start_latency_millis() > 86400000 {
    raise local_benchmark_error(LocalBenchmarkEvidence)
  }
  let measurements = result.measurements()
  let raw = measurements.raw_events_canonical_bytes()
  let summary_bytes = measurements.summary_canonical_bytes()
  if raw.is_empty() ||
    summary_bytes.is_empty() ||
    raw.length() > LOCAL_BENCHMARK_MAXIMUM_EVIDENCE_BYTES ||
    summary_bytes.length() >
    LOCAL_BENCHMARK_MAXIMUM_EVIDENCE_BYTES - raw.length() {
    raise local_benchmark_error(LocalBenchmarkEvidence)
  }
  let summary = measurements.summary()
  let serving = result.serving()
  if summary.submitted() != 1 ||
    summary.completed() != 1 ||
    summary.rejected() != 0 ||
    summary.timed_out() != 0 ||
    summary.cancelled() != 0 ||
    summary.failed() != 0 ||
    summary.input_tokens() != 1L ||
    summary.generated_tokens() != 2L ||
    serving.event_count() != 5 ||
    serving.network_accepts() != 1UL ||
    serving.network_disconnects() != 1UL {
    raise local_benchmark_error(LocalBenchmarkEvidence)
  }
  let canonical = @utf8.encode(
    "schema=lunaflux-local-runtime-benchmark.v1\n" +
    "runtime_recipe=" +
    release.runtime_recipe() +
    "\n" +
    "model_content_sha256=" +
    release.model_content_sha256() +
    "\n" +
    "model_plan_sha256=" +
    release.model_plan_sha256() +
    "\n" +
    "tokenizer_sha256=" +
    release.tokenizer_sha256() +
    "\n" +
    "worker_executable_sha256=" +
    release.worker_executable_sha256() +
    "\n" +
    "release_preflight_sha256=" +
    release.evidence_sha256() +
    "\n" +
    "workload=fixed-token-0,greedy,streaming,cache-disabled,input-1,output-2\n" +
    "profile=latency\ntrial_ordinal=1\norder_position=0\n" +
    "measurement_sha256=" +
    @benchmark_runner.system_monotonic_millis_measurement_digest().as_hex() +
    "\ncold_start_latency_millis=" +
    startup.cold_start_latency_millis().to_string() +
    "\nstartup_readiness=true\n" +
    "raw_events_sha256=" +
    measurements.raw_events_digest().as_hex() +
    "\nsummary_sha256=" +
    measurements.summary_digest().as_hex() +
    "\nraw_events_canonical_hex=" +
    @crypto.bytes_to_hex_string(raw) +
    "\nsummary_canonical_hex=" +
    @crypto.bytes_to_hex_string(summary_bytes) +
    "\nobserved_first_token=" +
    serving.first_token().to_string() +
    "\nobserved_second_token=" +
    serving.second_token().to_string() +
    "\nsubmitted=1\ncompleted=1\nrejected=0\ntimed_out=0\n" +
    "cancelled=0\nfailed=0\ninput_tokens=1\ngenerated_tokens=2\n" +
    "elapsed_ns=" +
    summary.elapsed_ns().to_string() +
    "\nqueue_ns=" +
    summary.queue_ns().total().to_string() +
    "\nttft_ns=" +
    summary.ttft_ns().total().to_string() +
    "\nservice_ns=" +
    summary.service_ns().total().to_string() +
    "\ndecode_ns=" +
    summary.decode_ns().total().to_string() +
    "\nend_to_end_ns=" +
    summary.end_to_end_ns().total().to_string() +
    "\nevent_order=accepted,token,token,usage,completed\nevent_count=5\n" +
    "network_accepts=1\nnetwork_disconnects=1\n" +
    "kv_pages_free_after_request=" +
    serving.kv_pages_free().to_string() +
    "\nterminal_phase=closed\nterminal_readiness=false\ncleanup_complete=true\n" +
    "comparison_admission=not-run\npromotion_authority=absent\n",
  )
  if canonical.length() > LOCAL_BENCHMARK_MAXIMUM_EVIDENCE_BYTES {
    raise local_benchmark_error(LocalBenchmarkEvidence)
  }
  {
    cold_start_latency_millis: startup.cold_start_latency_millis(),
    startup_readiness: startup.readiness(),
    first_token: serving.first_token(),
    second_token: serving.second_token(),
    raw_events: raw,
    summary: summary_bytes,
    canonical,
    digest: @benchmark_evidence.BenchmarkDigest::from_sha256(
      @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
    ) catch {
      _ => raise local_benchmark_error(LocalBenchmarkEvidence)
    },
  }
}

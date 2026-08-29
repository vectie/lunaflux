///|
const PHYSICAL_BENCHMARK_MAX_CANONICAL_BYTES : Int = 16384

///|
fn physical_benchmark_artifact_lengths_valid(
  raw_events_bytes : Int,
  summary_bytes : Int,
) -> Bool {
  raw_events_bytes > 0 &&
  summary_bytes > 0 &&
  raw_events_bytes <= PHYSICAL_BENCHMARK_MAX_CANONICAL_BYTES &&
  summary_bytes <= PHYSICAL_BENCHMARK_MAX_CANONICAL_BYTES &&
  raw_events_bytes <= PHYSICAL_BENCHMARK_MAX_CANONICAL_BYTES - summary_bytes
}

///|
fn encode_physical_benchmark_evidence(
  release : @runtime_instance.RuntimeReleaseAdmission,
  result : @runtime_instance.RuntimeSpawnedBenchmarkResult,
) -> String raise ApprovedModelSpawnedPhysicalError {
  let measurements = result.measurements()
  let raw = measurements.raw_events_canonical_bytes()
  let encoded_summary = measurements.summary_canonical_bytes()
  if !physical_benchmark_artifact_lengths_valid(
      raw.length(),
      encoded_summary.length(),
    ) {
    raise BenchmarkExecutionRejected
  }
  let summary = measurements.summary()
  let serving = result.serving()
  let evidence = "schema=lunaflux-physical-benchmark-trial.v1\n" +
    "engine=lunaflux\nprofile=latency\ntrial_ordinal=1\norder_position=0\n" +
    "workload_contract=pinned-1-input-2-output-qualification\n" +
    "timing_scope=native-listener-request-only\n" +
    "warmup_requests=0\nrequest_capacity=1\nclock_resolution_ns=1000000\n" +
    "upstream_revision=" +
    PINNED_UPSTREAM_REVISION +
    "\nruntime_recipe=" +
    release.runtime_recipe() +
    "\nmodel_content_sha256=" +
    release.model_content_sha256() +
    "\nmodel_plan_sha256=" +
    release.model_plan_sha256() +
    "\ntokenizer_sha256=" +
    release.tokenizer_sha256() +
    "\nworker_executable_sha256=" +
    release.worker_executable_sha256() +
    "\nrelease_preflight_sha256=" +
    release.evidence_sha256() +
    "\nmeasurement_sha256=" +
    @benchmark_runner.system_monotonic_millis_measurement_digest().as_hex() +
    "\nraw_events_sha256=" +
    measurements.raw_events_digest().as_hex() +
    "\nsummary_sha256=" +
    measurements.summary_digest().as_hex() +
    "\nraw_events_canonical_hex=" +
    @crypto.bytes_to_hex_string(raw) +
    "\nsummary_canonical_hex=" +
    @crypto.bytes_to_hex_string(encoded_summary) +
    "\nsubmitted=" +
    summary.submitted().to_string() +
    "\ncompleted=" +
    summary.completed().to_string() +
    "\nrejected=" +
    summary.rejected().to_string() +
    "\ntimed_out=" +
    summary.timed_out().to_string() +
    "\ncancelled=" +
    summary.cancelled().to_string() +
    "\nfailed=" +
    summary.failed().to_string() +
    "\ninput_tokens=" +
    summary.input_tokens().to_string() +
    "\ngenerated_tokens=" +
    summary.generated_tokens().to_string() +
    "\nelapsed_ns=" +
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
    "\nevent_order=accepted,token,token,usage,completed\n" +
    "event_count=" +
    serving.event_count().to_string() +
    "\nnetwork_accepts=" +
    serving.network_accepts().to_string() +
    "\nnetwork_disconnects=" +
    serving.network_disconnects().to_string() +
    "\nkv_pages_free_after_request=" +
    serving.kv_pages_free().to_string() +
    "\ncorrectness_scope=pinned-two-token-serving-contract\n" +
    "performance_scope=measured-qualification-not-baseline-comparison\n" +
    "comparison_admission=not-run\ncleanup_complete=1\n"
  evidence +
  "physical_benchmark_sha256=" +
  @crypto.bytes_to_hex_string(@crypto.sha256(@utf8.encode(evidence))) +
  "\n"
}

///|
async fn run_benchmark_campaign(
  deployment_argument : String,
  expected_worker_sha256 : String,
) -> Unit raise ApprovedModelSpawnedPhysicalError {
  let (release, owner) = prepare_pinned_campaign_owner(
    deployment_argument, expected_worker_sha256,
  )
  let handoff_matches = try owner.require_spawned_physical_handoff() catch {
    _ => false
  } noraise {
    _ => owner.readiness() == RuntimeNotReady
  }
  if !handoff_matches {
    ignore(drain_campaign_owner(owner) catch { _ => false })
    raise PhysicalHandoffRejected
  }
  let contract = @runtime_instance.RuntimeSpawnedBenchmarkContract::new(
    campaign_serving_validation_contract() catch {
      _ => {
        ignore(drain_campaign_owner(owner) catch { _ => false })
        raise BenchmarkExecutionRejected
      }
    },
    Latency,
    1,
    0,
  ) catch {
    _ => {
      ignore(drain_campaign_owner(owner) catch { _ => false })
      raise BenchmarkExecutionRejected
    }
  }
  let result = owner.benchmark_spawned_serving_request(contract) catch {
    @runtime_instance.RuntimeSpawnedBenchmarkFailed(rule) => {
      println("physical_benchmark_failure_rule=" + rule.code())
      raise BenchmarkExecutionRejected
    }
    _ => {
      println("physical_benchmark_failure_rule=unknown")
      raise BenchmarkExecutionRejected
    }
  }
  if owner.phase() != Closed ||
    owner.health() != RuntimeClosed ||
    owner.readiness() != RuntimeNotReady ||
    !owner.cleanup_complete() {
    raise CleanupIncomplete
  }
  println(encode_physical_benchmark_evidence(release, result))
}

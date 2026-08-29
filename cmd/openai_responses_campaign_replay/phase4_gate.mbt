///|
fn phase4_comparison(
  measurements : @openai_adapter.OpenAIResponsesComparisonMeasurementSet,
  correctness : ArrayView[@evidence.BenchmarkDigest],
) -> @evidence.BenchmarkComparison raise Phase4PrefixGateError {
  if measurements.trial_count() != PHASE4_CORRECTNESS_TRIALS ||
    correctness.length() != PHASE4_CORRECTNESS_TRIALS {
    raise Phase4InvalidCorrectness
  }
  let engines : Array[@evidence.BenchmarkEngineIdentity] = []
  for index in 0..<3 {
    engines.push(
      measurements.engine_identity(index) catch {
        _ => raise Phase4InvalidMeasurement
      },
    )
  }
  let workloads : Array[@evidence.BenchmarkWorkload] = []
  for index in 0..<9 {
    workloads.push(
      measurements.workload(index) catch {
        _ => raise Phase4InvalidMeasurement
      },
    )
  }
  let trials : Array[@evidence.BenchmarkTrial] = []
  for index in 0..<PHASE4_CORRECTNESS_TRIALS {
    trials.push(
      @evidence.BenchmarkTrial::from_summary(
        measurements=measurements.measurements(index) catch {
          _ => raise Phase4InvalidMeasurement
        },
        correctness_passed=true,
        correctness=correctness[index],
      ) catch {
        _ => raise Phase4InvalidCorrectness
      },
    )
  }
  @evidence.BenchmarkComparison::admit(
    environment=measurements.environment(),
    engines~,
    workloads~,
    trials~,
  ) catch {
    _ => raise Phase4InvalidCorrectness
  }
}

///|
fn phase4_gate_evidence(
  measurements : @openai_adapter.OpenAIResponsesComparisonMeasurementSet,
  correctness : ArrayView[@evidence.BenchmarkDigest],
  policy : Phase4ExternalPolicy,
) -> Phase4PrefixGateEvidence raise Phase4PrefixGateError {
  let cold_regression_budget_bps = policy.cold_regression_budget_bps
  if cold_regression_budget_bps < 0 ||
    cold_regression_budget_bps > PHASE4_MAX_COLD_REGRESSION_BPS {
    raise Phase4InvalidPolicy
  }
  let comparison = phase4_comparison(measurements, correctness)
  let rich = phase4_profile_matrix(measurements, 4)
  let cold = phase4_profile_matrix(measurements, 5)
  let (rich_baseline_name, rich_baseline) = phase4_best_baseline(rich)
  let (cold_baseline_name, cold_baseline) = phase4_best_baseline(cold)
  let rich_passed = phase4_rate_at_least(rich.lunaflux, rich_baseline)
  let cold_passed = phase4_rate_within_budget(
    cold.lunaflux,
    cold_baseline,
    cold_regression_budget_bps,
  )
  let passed = rich_passed && cold_passed
  let output = StringBuilder()
  output.write_string("schema=lunaflux-phase4-prefix-benchmark-gate.v1\n")
  output.write_string("phase4_policy_sha256=" + policy.digest + "\n")
  output.write_string(
    "phase4_gate_executable_sha256=" + policy.gate_executable + "\n",
  )
  output.write_string(
    "phase4_fd_exec_helper_sha256=" + policy.fd_exec_helper + "\n",
  )
  output.write_string(
    "comparison_sha256=" + comparison.digest().as_hex() + "\n",
  )
  output.write_string(
    "capture_inventory_sha256=" + policy.capture_inventory + "\n",
  )
  output.write_string(
    "correctness_inventory_sha256=" + policy.correctness_inventory + "\n",
  )
  output.write_string("declaration_sha256=" + policy.declaration + "\n")
  output.write_string(
    "prefix_rich_workload_sha256=" + policy.prefix_rich_workload + "\n",
  )
  output.write_string(
    "prefix_cold_workload_sha256=" + policy.prefix_cold_workload + "\n",
  )
  output.write_string("campaign_tool_sha256=" + policy.campaign_tool + "\n")
  output.write_string(
    "correctness_verifier_sha256=" + policy.correctness_verifier + "\n",
  )
  output.write_string(
    "engine_identity_verifier_sha256=" + policy.identity_verifier + "\n",
  )
  output.write_string(
    "process_supervisor_sha256=" + policy.process_supervisor + "\n",
  )
  output.write_string("trial_matrix=3-engines,9-profiles,3-trials\n")
  output.write_string("profile_trial_count=9\n")
  output.write_string("counterbalance=latin-square-v1\n")
  output.write_string("observation_protocol=openai.responses.sse.v1\n")
  output.write_string("metric=aggregate_completed_requests_per_nanosecond\n")
  output.write_string(
    "cold_regression_budget_bps=" +
    cold_regression_budget_bps.to_string() +
    "\n",
  )
  phase4_append_aggregate(output, "prefix_rich", "lunaflux", rich.lunaflux)
  phase4_append_aggregate(output, "prefix_rich", "vllm", rich.vllm)
  phase4_append_aggregate(output, "prefix_rich", "sglang", rich.sglang)
  phase4_append_aggregate(output, "prefix_cold", "lunaflux", cold.lunaflux)
  phase4_append_aggregate(output, "prefix_cold", "vllm", cold.vllm)
  phase4_append_aggregate(output, "prefix_cold", "sglang", cold.sglang)
  output.write_string("prefix_rich_best_baseline=" + rich_baseline_name + "\n")
  output.write_string(
    "prefix_rich_parity=" + (if rich_passed { "pass\n" } else { "fail\n" }),
  )
  output.write_string("prefix_cold_best_baseline=" + cold_baseline_name + "\n")
  output.write_string(
    "prefix_cold_regression=" + (if cold_passed { "pass\n" } else { "fail\n" }),
  )
  output.write_string("correctness_join=external-artifact-matrix\n")
  output.write_string("external_verifier_required=1\n")
  output.write_string("physical_measurement_claim=none\n")
  output.write_string("comparison_authority=none\n")
  output.write_string("release_authority=none\n")
  output.write_string(
    "phase4_prefix_gate=" + (if passed { "pass\n" } else { "fail\n" }),
  )
  let canonical = @utf8.encode(output.to_string())
  let digest = @evidence.BenchmarkDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
  ) catch {
    _ => raise Phase4InvalidMeasurement
  }
  { canonical, digest, passed }
}

///|
fn fused_benchmark_bool(value : Bool) -> String {
  if value {
    "1"
  } else {
    "0"
  }
}

///|
fn fused_benchmark_order_name(
  order : FusedCandidateBenchmarkPairOrder,
) -> String {
  match order {
    BaselineThenCandidate => "baseline-then-candidate"
    CandidateThenBaseline => "candidate-then-baseline"
  }
}

///|
fn fused_benchmark_append_identity(
  output : StringBuilder,
  identity : FusedBenchmarkCandidateIdentity,
) -> Unit {
  output.write_string("family=" + identity.family_name + "\n")
  output.write_string("candidate_sha256=" + identity.candidate.as_hex() + "\n")
  output.write_string("source_sha256=" + identity.source.as_hex() + "\n")
  output.write_string("recipe_sha256=" + identity.recipe.as_hex() + "\n")
  output.write_string(
    "target=sm_" +
    identity.target_compute_major.to_string() +
    identity.target_compute_minor.to_string() +
    "\n",
  )
  output.write_string(
    "target_bf16=" + fused_benchmark_bool(identity.target_bf16) + "\n",
  )
  output.write_string(
    "target_cublas_lt=" + fused_benchmark_bool(identity.target_cublas_lt) + "\n",
  )
  output.write_string("profile_id=" + identity.profile_id.to_string() + "\n")
  output.write_string(
    "max_query_rows=" + identity.max_query_rows.to_string() + "\n",
  )
  output.write_string(
    "max_query_tokens=" + identity.max_query_tokens.to_string() + "\n",
  )
  output.write_string(
    "max_page_table_entries=" +
    identity.max_page_table_entries.to_string() +
    "\n",
  )
  output.write_string(
    "launch=" +
    identity.grid_x.to_string() +
    ":" +
    identity.grid_y.to_string() +
    ":" +
    identity.grid_z.to_string() +
    "," +
    identity.block_x.to_string() +
    ":" +
    identity.block_y.to_string() +
    ":" +
    identity.block_z.to_string() +
    ",shared=" +
    identity.shared_memory_bytes.to_string() +
    "\n",
  )
  output.write_string("toolchain_sha256=" + identity.toolchain + "\n")
  output.write_string(
    "compiler_version=" +
    identity.compiler_major.to_string() +
    "." +
    identity.compiler_minor.to_string() +
    "." +
    identity.compiler_patch.to_string() +
    "\n",
  )
  output.write_string(
    "compiler_policy=opt" +
    identity.optimization_level.to_string() +
    ",fmad=" +
    fused_benchmark_bool(identity.fused_multiply_add) +
    ",reassociate=" +
    fused_benchmark_bool(identity.reassociate) +
    ",max_registers=" +
    identity.max_registers.to_string() +
    "\n",
  )
  output.write_string("numerical_policy=" + identity.numerical_policy + "\n")
  output.write_string("function_symbol=" + identity.function_symbol + "\n")
  output.write_string(
    "dispatch_canary=" + identity.dispatch_canary.to_string() + "\n",
  )
  output.write_string(identity.fallback_lines)
}

///|
fn fused_benchmark_append_arm(
  output : StringBuilder,
  arm : FusedCandidateBenchmarkArmObservation,
) -> Unit {
  output.write_string(arm.duration_ns.to_string())
  output.write_string(",")
  output.write_string(arm.work_units.to_string())
  output.write_string(",")
  output.write_string(arm.resources_acquired.to_string())
  output.write_string(",")
  output.write_string(arm.resources_released.to_string())
  output.write_string(",")
  output.write_string(arm.live_resources_before.to_string())
  output.write_string(",")
  output.write_string(arm.live_resources_after.to_string())
  output.write_string(",")
  output.write_string(arm.live_bytes_before.to_string())
  output.write_string(",")
  output.write_string(arm.live_bytes_after.to_string())
  output.write_string(",")
  output.write_string(arm.resource_failures.to_string())
  output.write_string(",")
  output.write_string(arm.dispatch_canary_before.to_string())
  output.write_string(",")
  output.write_string(arm.dispatch_canary_after.to_string())
}

///|
fn fused_benchmark_append_observations(
  output : StringBuilder,
  observations : ArrayView[FusedCandidateBenchmarkPairedObservation],
) -> Unit {
  for observation in observations {
    output.write_string("observation=")
    output.write_string(
      fused_benchmark_shape_ordinal(observation.shape).to_string(),
    )
    output.write_string(",")
    output.write_string(observation.trial_ordinal.to_string())
    output.write_string(",")
    output.write_string(fused_benchmark_order_name(observation.order))
    output.write_string(",repetitions=")
    output.write_string(observation.repetitions.to_string())
    output.write_string(",baseline=")
    fused_benchmark_append_arm(output, observation.baseline)
    output.write_string(",candidate=")
    fused_benchmark_append_arm(output, observation.candidate)
    output.write_string("\n")
  }
}

///|
fn fused_benchmark_append_summaries(
  output : StringBuilder,
  summaries : ArrayView[FusedBenchmarkShapeSummary],
) -> Unit {
  for summary in summaries {
    let name = fused_benchmark_shape_name(summary.shape)
    output.write_string(
      fused_benchmark_summary_line(
        "summary=" + name + ",baseline",
        summary.baseline,
      ),
    )
    output.write_string(
      fused_benchmark_summary_line(
        "summary=" + name + ",candidate",
        summary.candidate,
      ),
    )
    output.write_string(
      "work_summary=" +
      name +
      ",baseline_total=" +
      summary.baseline_work_total.to_string() +
      ",candidate_total=" +
      summary.candidate_work_total.to_string() +
      "\n",
    )
  }
}

///|
fn fused_benchmark_canonical(
  identity : FusedBenchmarkCandidateIdentity,
  protocol : FusedCandidateBenchmarkProtocol,
  observations : ArrayView[FusedCandidateBenchmarkPairedObservation],
  summaries : ArrayView[FusedBenchmarkShapeSummary],
) -> Bytes {
  let output = StringBuilder()
  output.write_string(
    "schema=lunaflux-fused-candidate-benchmark-qualification.v1\n",
  )
  fused_benchmark_append_identity(output, identity)
  output.write_string("shape_set_sha256=" + protocol.shape_set.as_hex() + "\n")
  output.write_string("clock_sha256=" + protocol.clock.as_hex() + "\n")
  output.write_string(
    "resource_collector_sha256=" + protocol.resource_collector.as_hex() + "\n",
  )
  output.write_string(
    "warmup_iterations=" + protocol.warmup_iterations.to_string() + "\n",
  )
  output.write_string(
    "trials_per_shape=" + protocol.trials_per_shape.to_string() + "\n",
  )
  output.write_string(
    "repetitions_per_arm=" + protocol.repetitions_per_arm.to_string() + "\n",
  )
  output.write_string("pair_order=alternating-baseline-candidate.v1\n")
  output.write_string(
    "win_policy=every-pair-strict-and-each-shape-total-at-least-5-percent.v1\n",
  )
  output.write_string("shape_count=4\n")
  output.write_string(
    "observation_count=" + observations.length().to_string() + "\n",
  )
  fused_benchmark_append_observations(output, observations)
  fused_benchmark_append_summaries(output, summaries)
  output.write_string(
    "measurement_authority=caller-supplied-raw-observations\n",
  )
  output.write_string("physical_reproduction_authority=external\n")
  output.write_string("mixed_workload_authority=absent\n")
  output.write_string("manifest_bindable=false\n")
  output.write_string("promotion_authority=false\n")
  @utf8.encode(output.to_string())
}

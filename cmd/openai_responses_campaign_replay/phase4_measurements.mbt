///|
fn phase4_require_successful_summary(
  summary : @evidence.BenchmarkTrialSummary,
) -> Unit raise Phase4PrefixGateError {
  if summary.submitted() <= 0 ||
    summary.completed() != summary.submitted() ||
    summary.rejected() != 0 ||
    summary.timed_out() != 0 ||
    summary.cancelled() != 0 ||
    summary.failed() != 0 ||
    summary.generated_tokens() <= 0L ||
    summary.generated_tokens() > PHASE4_MAX_TRIAL_GENERATED_TOKENS ||
    summary.elapsed_ns() <= 0L ||
    summary.elapsed_ns() > PHASE4_MAX_TRIAL_ELAPSED_NS {
    raise Phase4InvalidMeasurement
  }
}

///|
fn phase4_require_equivalent_work(
  reference : @evidence.BenchmarkTrialSummary,
  candidate : @evidence.BenchmarkTrialSummary,
) -> Unit raise Phase4PrefixGateError {
  if candidate.submitted() != reference.submitted() ||
    candidate.completed() != reference.completed() ||
    candidate.input_tokens() != reference.input_tokens() ||
    candidate.generated_tokens() != reference.generated_tokens() {
    raise Phase4InvalidMeasurement
  }
}

///|
fn phase4_profile_aggregate(
  measurements : @openai_adapter.OpenAIResponsesComparisonMeasurementSet,
  profile_index : Int,
  engine_index : Int,
) -> Phase4ProfileAggregate raise Phase4PrefixGateError {
  let mut completed = 0L
  let mut input_tokens = 0L
  let mut generated_tokens = 0L
  let mut elapsed_ns = 0L
  let mut ttft_ns = 0L
  let mut end_to_end_ns = 0L
  for ordinal_index in 0..<3 {
    let base = profile_index * PHASE4_PROFILE_TRIALS + ordinal_index * 3
    let reference = measurements.measurements(base).summary() catch {
        _ => raise Phase4InvalidMeasurement
      }
    let summary = measurements.measurements(base + engine_index).summary() catch {
        _ => raise Phase4InvalidMeasurement
      }
    phase4_require_successful_summary(reference)
    phase4_require_successful_summary(summary)
    phase4_require_equivalent_work(reference, summary)
    completed = phase4_checked_add(
      completed,
      Int64::from_int(summary.completed()),
    )
    input_tokens = phase4_checked_add(input_tokens, summary.input_tokens())
    generated_tokens = phase4_checked_add(
      generated_tokens,
      summary.generated_tokens(),
    )
    elapsed_ns = phase4_checked_add(elapsed_ns, summary.elapsed_ns())
    ttft_ns = phase4_checked_add(ttft_ns, summary.ttft_ns().total())
    end_to_end_ns = phase4_checked_add(
      end_to_end_ns,
      summary.end_to_end_ns().total(),
    )
  }
  {
    completed,
    input_tokens,
    generated_tokens,
    elapsed_ns,
    ttft_ns,
    end_to_end_ns,
  }
}

///|
fn phase4_profile_matrix(
  measurements : @openai_adapter.OpenAIResponsesComparisonMeasurementSet,
  profile_index : Int,
) -> Phase4ProfileMatrix raise Phase4PrefixGateError {
  {
    lunaflux: phase4_profile_aggregate(measurements, profile_index, 0),
    vllm: phase4_profile_aggregate(measurements, profile_index, 1),
    sglang: phase4_profile_aggregate(measurements, profile_index, 2),
  }
}

///|
fn phase4_append_aggregate(
  output : StringBuilder,
  profile : String,
  engine : String,
  aggregate : Phase4ProfileAggregate,
) -> Unit {
  let prefix = profile + "_" + engine + "_"
  output.write_string(
    prefix + "completed=" + aggregate.completed.to_string() + "\n",
  )
  output.write_string(
    prefix + "input_tokens=" + aggregate.input_tokens.to_string() + "\n",
  )
  output.write_string(
    prefix + "generated_tokens=" + aggregate.generated_tokens.to_string() + "\n",
  )
  output.write_string(
    prefix + "elapsed_ns=" + aggregate.elapsed_ns.to_string() + "\n",
  )
  output.write_string(
    prefix + "ttft_total_ns=" + aggregate.ttft_ns.to_string() + "\n",
  )
  output.write_string(
    prefix + "end_to_end_total_ns=" + aggregate.end_to_end_ns.to_string() + "\n",
  )
  output.write_string(
    prefix +
    "completed_rate_fraction=" +
    aggregate.completed.to_string() +
    "/" +
    aggregate.elapsed_ns.to_string() +
    "\n",
  )
}

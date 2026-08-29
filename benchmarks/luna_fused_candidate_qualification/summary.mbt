///|
priv struct FusedBenchmarkShapeSummary {
  shape : FusedCandidateBenchmarkShape
  baseline : FusedBenchmarkIntegerSummary
  candidate : FusedBenchmarkIntegerSummary
  baseline_work_total : Int64
  candidate_work_total : Int64
}

///|
fn fused_benchmark_checked_add(
  left : Int64,
  right : Int64,
  stage : FusedCandidateBenchmarkStage,
) -> Int64 raise FusedCandidateBenchmarkError {
  if right < 0L || left > 9223372036854775807L - right {
    raise OverflowedFusedCandidateBenchmark(stage)
  }
  left + right
}

///|
fn fused_benchmark_summarize(
  values : Array[Int64],
) -> FusedBenchmarkIntegerSummary raise FusedCandidateBenchmarkError {
  if values.is_empty() {
    raise InvalidFusedCandidateBenchmark(DurationSummary)
  }
  let sorted = values.copy()
  sorted.sort()
  let mut total = 0L
  for value in sorted {
    if value <= 0L {
      raise InvalidFusedCandidateBenchmark(DurationSummary)
    }
    total = fused_benchmark_checked_add(total, value, DurationSummary)
  }
  let p50_index = (sorted.length() * 50 + 99) / 100 - 1
  {
    count: sorted.length(),
    minimum: sorted[0],
    p50: sorted[p50_index],
    maximum: sorted[sorted.length() - 1],
    total,
  }
}

///|
fn fused_benchmark_summary_line(
  label : String,
  summary : FusedBenchmarkIntegerSummary,
) -> String {
  label +
  "_count=" +
  summary.count.to_string() +
  ",min_ns=" +
  summary.minimum.to_string() +
  ",p50_ns=" +
  summary.p50.to_string() +
  ",max_ns=" +
  summary.maximum.to_string() +
  ",total_ns=" +
  summary.total.to_string() +
  "\n"
}

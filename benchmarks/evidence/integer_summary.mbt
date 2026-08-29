///|
const BENCHMARK_MAX_REQUEST_EVENTS : Int = 8192

///|
const BENCHMARK_MAX_INT64 : Int64 = 9223372036854775807L

///|
/// The hard request-event bound for one in-memory summary admission.
pub fn maximum_request_events_per_summary() -> Int {
  BENCHMARK_MAX_REQUEST_EVENTS
}

///|
/// A deterministic nearest-rank integer distribution. Empty observation sets
/// have count and all scalars equal to zero.
pub struct BenchmarkIntegerSummary {
  priv count : Int
  priv minimum : Int64
  priv p50 : Int64
  priv p95 : Int64
  priv p99 : Int64
  priv maximum : Int64
  priv total : Int64
}

///|
fn checked_summary_add(
  left : Int64,
  right : Int64,
) -> Int64 raise BenchmarkEvidenceError {
  if left < 0L || right < 0L || left > BENCHMARK_MAX_INT64 - right {
    raise SummaryIntegerOverflow
  }
  left + right
}

///|
fn percentile_index(length : Int, percentile : Int) -> Int {
  (length * percentile + 99) / 100 - 1
}

///|
fn summarize_integers(
  observations : Array[Int64],
) -> BenchmarkIntegerSummary raise BenchmarkEvidenceError {
  if observations.is_empty() {
    return {
      count: 0,
      minimum: 0L,
      p50: 0L,
      p95: 0L,
      p99: 0L,
      maximum: 0L,
      total: 0L,
    }
  }
  let mut total = 0L
  for value in observations {
    total = checked_summary_add(total, value)
  }
  observations.sort()
  let length = observations.length()
  {
    count: length,
    minimum: observations[0],
    p50: observations[percentile_index(length, 50)],
    p95: observations[percentile_index(length, 95)],
    p99: observations[percentile_index(length, 99)],
    maximum: observations[length - 1],
    total,
  }
}

///|
pub fn BenchmarkIntegerSummary::count(self : BenchmarkIntegerSummary) -> Int {
  self.count
}

///|
pub fn BenchmarkIntegerSummary::minimum(
  self : BenchmarkIntegerSummary,
) -> Int64 {
  self.minimum
}

///|
pub fn BenchmarkIntegerSummary::p50(self : BenchmarkIntegerSummary) -> Int64 {
  self.p50
}

///|
pub fn BenchmarkIntegerSummary::p95(self : BenchmarkIntegerSummary) -> Int64 {
  self.p95
}

///|
pub fn BenchmarkIntegerSummary::p99(self : BenchmarkIntegerSummary) -> Int64 {
  self.p99
}

///|
pub fn BenchmarkIntegerSummary::maximum(
  self : BenchmarkIntegerSummary,
) -> Int64 {
  self.maximum
}

///|
pub fn BenchmarkIntegerSummary::total(self : BenchmarkIntegerSummary) -> Int64 {
  self.total
}

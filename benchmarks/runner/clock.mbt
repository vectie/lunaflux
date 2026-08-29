///|
const BENCHMARK_RUNNER_MAX_INT64 : Int64 = 9223372036854775807L

///|
const BENCHMARK_RUNNER_MAX_CLOCK_MILLIS : UInt64 = 9223372036854UL

///|
const BENCHMARK_RUNNER_NANOS_PER_MILLI : Int64 = 1000000L

///|
/// The digest of the exact collection schema
/// `lunaflux.benchmark-runner.system-monotonic-milliseconds.v1`. Evidence
/// timestamps use nanosecond units but have truthful one-millisecond
/// resolution.
pub fn system_monotonic_millis_measurement_digest() -> @evidence.BenchmarkDigest {
  try! @evidence.BenchmarkDigest::from_sha256(
    "db8b9983e81661e4c955ebaed2edf6c35ba789529fa88b2f02f8b7471ca2dd7b",
  )
}

///|
/// The distinct measurement identity used when an external live runner owns
/// timestamps and a later process replays them. This must never be confused
/// with the in-process system monotonic clock identity above.
pub fn captured_monotonic_nanos_measurement_digest() -> @evidence.BenchmarkDigest {
  try! @evidence.BenchmarkDigest::from_sha256(
    "5b0872f4a1379903204d1bf61b5f51a80af28c7c559634abcc1a988467ce4d4b",
  )
}

///|
pub fn CapturedMonotonicTimestamp::from_nanoseconds(
  nanoseconds : Int64,
) -> CapturedMonotonicTimestamp raise BenchmarkRunnerError {
  if nanoseconds < 0L {
    raise BenchmarkRunnerClockOutOfRange
  }
  { nanoseconds, }
}

///|
fn checked_nonnegative_add(
  left : Int64,
  right : Int64,
) -> Int64 raise BenchmarkRunnerError {
  if left < 0L || right < 0L || left > BENCHMARK_RUNNER_MAX_INT64 - right {
    raise BenchmarkRunnerIntegerOverflow
  }
  left + right
}

///|
fn BenchmarkTrialCollector::read_timestamp_ns(
  self : BenchmarkTrialCollector,
) -> Int64 raise BenchmarkRunnerError {
  let millis = self.clock.now_millis() catch {
    _ => raise BenchmarkRunnerClockUnavailable
  }
  if millis > BENCHMARK_RUNNER_MAX_CLOCK_MILLIS {
    raise BenchmarkRunnerClockOutOfRange
  }
  millis.reinterpret_as_int64() * BENCHMARK_RUNNER_NANOS_PER_MILLI
}

///|
fn BenchmarkTrialCollector::validate_timestamp(
  self : BenchmarkTrialCollector,
  timestamp_ns : Int64,
) -> Unit raise BenchmarkRunnerError {
  if timestamp_ns < 0L {
    raise BenchmarkRunnerClockOutOfRange
  }
  match self.last_timestamp_ns {
    Some(previous) =>
      if timestamp_ns < previous {
        raise BenchmarkRunnerClockMovedBackward
      }
    None => ()
  }
}

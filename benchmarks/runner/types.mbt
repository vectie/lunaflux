///|
/// Payload-free failures from the bounded benchmark lifecycle collector.
pub(all) suberror BenchmarkRunnerError {
  BenchmarkRunnerInvalidCapacity
  BenchmarkRunnerInvalidTrial
  BenchmarkRunnerInvalidRequest
  BenchmarkRunnerCapacityExhausted
  BenchmarkRunnerDuplicateRequest
  BenchmarkRunnerNonCanonicalRequestOrder
  BenchmarkRunnerInvalidLifecycle
  BenchmarkRunnerClockUnavailable
  BenchmarkRunnerClockOutOfRange
  BenchmarkRunnerClockMovedBackward
  BenchmarkRunnerIntegerOverflow
  BenchmarkRunnerIncompleteTrial
  BenchmarkRunnerAlreadyFinished
  BenchmarkRunnerEvidenceRejected
} derive(Debug, Eq)

///|
enum BenchmarkRequestState {
  Vacant
  Submitted(Int64, Int64)
  Admitted(Int64, Int64, Int64)
  Streaming(Int64, Int64, Int64, Int64)
  Terminal(@evidence.BenchmarkRequestEvent)
}

///|
/// A startup-bounded, single-owner request lifecycle collector for one
/// engine/profile trial. Its public observations sample the process monotonic
/// clock at millisecond resolution and encode those samples as nanoseconds.
/// It owns no process, filesystem, network, device, or baseline authority.
pub struct BenchmarkTrialCollector {
  priv engine : @evidence.BenchmarkEngine
  priv profile : @evidence.BenchmarkProfile
  priv ordinal : Int32
  priv order_position : Int32
  priv request_capacity : Int32
  priv slots : FixedArray[BenchmarkRequestState]
  priv mut submitted : Int32
  priv mut terminal : Int32
  priv mut input_tokens : Int64
  priv mut generated_tokens : Int64
  priv mut last_timestamp_ns : Int64?
  priv mut finished : Bool
  priv captured_mode : Bool
  priv measurement : @evidence.BenchmarkDigest
  priv clock : @monotonic_clock.MonotonicClock
}

///|
/// A monotonic timestamp captured by the external owner of a replay artifact.
/// Construction validates only its numeric domain; it does not claim that the
/// current process sampled the clock or owned the original transport.
pub struct CapturedMonotonicTimestamp {
  priv nanoseconds : Int64
} derive(Debug, Eq)

///|
pub extend BenchmarkRunnerError with @debug.Debug::{to_repr}

///|
pub extend BenchmarkRunnerError with Eq::{not_equal, equal}

///|
pub extend CapturedMonotonicTimestamp with @debug.Debug::{to_repr}

///|
pub extend CapturedMonotonicTimestamp with Eq::{not_equal, equal}

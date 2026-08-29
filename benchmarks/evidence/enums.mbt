///|
/// The three engines required by the public comparison protocol.
pub(all) enum BenchmarkEngine {
  LunaFlux
  Vllm
  Sglang
} derive(Debug, Eq)

///|
/// The complete workload envelope. A comparison cannot omit a profile where
/// an engine loses or fails.
pub(all) enum BenchmarkProfile {
  Latency
  Chat
  LongPrefill
  DecodeHeavy
  PrefixRich
  PrefixCold
  Saturation
  Churn
  Mixed
} derive(Debug, Eq)

///|
/// The terminal outcome of one submitted request. Every raw request event has
/// exactly one outcome, so summaries cannot omit rejected or failed traffic.
pub(all) enum BenchmarkRequestOutcome {
  Completed
  Rejected
  TimedOut
  Cancelled
  Failed
} derive(Debug, Eq)

///|
/// Stable, payload-free comparison admission failures.
pub(all) suberror BenchmarkEvidenceError {
  InvalidDigest
  InvalidEngineIdentity
  InvalidWorkload
  InvalidTrial
  IncorrectOutcomeAccounting
  CorrectnessNotEstablished
  InvalidComparisonShape
  DuplicateTrial
  BiasedEngineOrder
  CanonicalEvidenceTooLarge
  InvalidRequestEvent
  InvalidEventTimestamp
  NonCanonicalEventOrder
  BenchmarkEventLimitExceeded
  SummaryIntegerOverflow
} derive(Debug, Eq)

///|
fn engine_tag(engine : BenchmarkEngine) -> Int {
  match engine {
    LunaFlux => 0
    Vllm => 1
    Sglang => 2
  }
}

///|
fn profile_tag(profile : BenchmarkProfile) -> Int {
  match profile {
    Latency => 0
    Chat => 1
    LongPrefill => 2
    DecodeHeavy => 3
    PrefixRich => 4
    PrefixCold => 5
    Saturation => 6
    Churn => 7
    Mixed => 8
  }
}

///|
fn request_outcome_tag(outcome : BenchmarkRequestOutcome) -> Int {
  match outcome {
    Completed => 0
    Rejected => 1
    TimedOut => 2
    Cancelled => 3
    Failed => 4
  }
}

///|
fn all_profiles() -> Array[BenchmarkProfile] {
  [
    Latency,
    Chat,
    LongPrefill,
    DecodeHeavy,
    PrefixRich,
    PrefixCold,
    Saturation,
    Churn,
    Mixed,
  ]
}

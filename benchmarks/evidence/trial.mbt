///|
/// One measured engine/profile trial. All request outcomes are retained; raw
/// events, correctness evidence, and the computed summary are independently
/// digest-bound.
pub struct BenchmarkTrial {
  priv engine : BenchmarkEngine
  priv profile : BenchmarkProfile
  priv ordinal : Int
  priv order_position : Int
  priv submitted : Int
  priv completed : Int
  priv rejected : Int
  priv timed_out : Int
  priv cancelled : Int
  priv failed : Int
  priv correctness_passed : Bool
  priv raw_events : BenchmarkDigest
  priv correctness : BenchmarkDigest
  priv summary : BenchmarkDigest
}

///|
pub fn BenchmarkTrial::new(
  engine~ : BenchmarkEngine,
  profile~ : BenchmarkProfile,
  ordinal~ : Int,
  order_position~ : Int,
  submitted~ : Int,
  completed~ : Int,
  rejected~ : Int,
  timed_out~ : Int,
  cancelled~ : Int,
  failed~ : Int,
  correctness_passed~ : Bool,
  raw_events~ : BenchmarkDigest,
  correctness~ : BenchmarkDigest,
  summary~ : BenchmarkDigest,
) -> BenchmarkTrial raise BenchmarkEvidenceError {
  if ordinal < 1 || ordinal > 3 || order_position < 0 || order_position > 2 {
    raise InvalidTrial
  }
  if submitted <= 0 ||
    submitted > 1000000000 ||
    completed < 0 ||
    completed > submitted ||
    rejected < 0 ||
    rejected > submitted ||
    timed_out < 0 ||
    timed_out > submitted ||
    cancelled < 0 ||
    cancelled > submitted ||
    failed < 0 ||
    failed > submitted {
    raise InvalidTrial
  }
  let accounted = Int64::from_int(completed) +
    Int64::from_int(rejected) +
    Int64::from_int(timed_out) +
    Int64::from_int(cancelled) +
    Int64::from_int(failed)
  if accounted != Int64::from_int(submitted) {
    raise IncorrectOutcomeAccounting
  }
  if !correctness_passed {
    raise CorrectnessNotEstablished
  }
  if raw_events == correctness ||
    raw_events == summary ||
    correctness == summary {
    raise InvalidTrial
  }
  {
    engine,
    profile,
    ordinal,
    order_position,
    submitted,
    completed,
    rejected,
    timed_out,
    cancelled,
    failed,
    correctness_passed,
    raw_events,
    correctness,
    summary,
  }
}

///|
/// Builds a trial from summary evidence whose identity, counts, raw-event
/// digest, and summary digest were admitted together. This avoids rebinding a
/// summary to a different engine/profile trial at the call site.
pub fn BenchmarkTrial::from_summary(
  measurements~ : BenchmarkSummaryEvidence,
  correctness_passed~ : Bool,
  correctness~ : BenchmarkDigest,
) -> BenchmarkTrial raise BenchmarkEvidenceError {
  let summary = measurements.summary
  BenchmarkTrial::new(
    engine=measurements.engine,
    profile=measurements.profile,
    ordinal=measurements.ordinal,
    order_position=measurements.order_position,
    submitted=summary.submitted,
    completed=summary.completed,
    rejected=summary.rejected,
    timed_out=summary.timed_out,
    cancelled=summary.cancelled,
    failed=summary.failed,
    correctness_passed~,
    raw_events=measurements.raw_events_digest,
    correctness~,
    summary=measurements.summary_digest,
  )
}

///|
pub fn BenchmarkTrial::engine(self : BenchmarkTrial) -> BenchmarkEngine {
  self.engine
}

///|
pub fn BenchmarkTrial::profile(self : BenchmarkTrial) -> BenchmarkProfile {
  self.profile
}

///|
pub fn BenchmarkTrial::ordinal(self : BenchmarkTrial) -> Int {
  self.ordinal
}

///|
pub fn BenchmarkTrial::order_position(self : BenchmarkTrial) -> Int {
  self.order_position
}

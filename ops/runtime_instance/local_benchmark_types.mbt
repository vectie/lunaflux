///|
pub(all) enum RuntimeLocalBenchmarkRule {
  LocalBenchmarkRelease
  LocalBenchmarkContract
  LocalBenchmarkRuntime
  LocalBenchmarkEvidence
  LocalBenchmarkCleanup
} derive(Debug, Eq)

///|
pub(all) suberror RuntimeLocalBenchmarkError {
  RuntimeLocalBenchmarkFailed(RuntimeLocalBenchmarkRule)
} derive(Debug, Eq)

///|
fn local_benchmark_error(
  rule : RuntimeLocalBenchmarkRule,
) -> RuntimeLocalBenchmarkError {
  RuntimeLocalBenchmarkFailed(rule)
}

///|
/// Root-free raw measurement evidence for one fixed authenticated local
/// request. It contains no comparison, manifest, or promotion authority.
pub struct RuntimeLocalBenchmarkEvidence {
  priv cold_start_latency_millis : Int
  priv startup_readiness : RuntimeInstanceReadiness
  priv first_token : Int
  priv second_token : Int
  priv raw_events : Bytes
  priv summary : Bytes
  priv canonical : Bytes
  priv digest : @benchmark_evidence.BenchmarkDigest
}

///|
pub fn RuntimeLocalBenchmarkEvidence::cold_start_latency_millis(
  self : RuntimeLocalBenchmarkEvidence,
) -> Int {
  self.cold_start_latency_millis
}

///|
pub fn RuntimeLocalBenchmarkEvidence::startup_readiness(
  self : RuntimeLocalBenchmarkEvidence,
) -> RuntimeInstanceReadiness {
  self.startup_readiness
}

///|
pub fn RuntimeLocalBenchmarkEvidence::first_token(
  self : RuntimeLocalBenchmarkEvidence,
) -> Int {
  self.first_token
}

///|
pub fn RuntimeLocalBenchmarkEvidence::second_token(
  self : RuntimeLocalBenchmarkEvidence,
) -> Int {
  self.second_token
}

///|
pub fn RuntimeLocalBenchmarkEvidence::raw_events_canonical_bytes(
  self : RuntimeLocalBenchmarkEvidence,
) -> Bytes {
  self.raw_events
}

///|
pub fn RuntimeLocalBenchmarkEvidence::summary_canonical_bytes(
  self : RuntimeLocalBenchmarkEvidence,
) -> Bytes {
  self.summary
}

///|
pub fn RuntimeLocalBenchmarkEvidence::canonical_bytes(
  self : RuntimeLocalBenchmarkEvidence,
) -> Bytes {
  self.canonical
}

///|
pub fn RuntimeLocalBenchmarkEvidence::digest(
  self : RuntimeLocalBenchmarkEvidence,
) -> @benchmark_evidence.BenchmarkDigest {
  self.digest
}

///|
pub fn RuntimeLocalBenchmarkEvidence::promotion_authorized(
  _self : RuntimeLocalBenchmarkEvidence,
) -> Bool {
  false
}

///|
pub fn RuntimeLocalBenchmarkEvidence::comparison_admitted(
  _self : RuntimeLocalBenchmarkEvidence,
) -> Bool {
  false
}

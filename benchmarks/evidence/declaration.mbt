///|
/// Immutable pins shared by every engine and trial in one comparison.
pub struct BenchmarkEnvironment {
  priv hardware : BenchmarkDigest
  priv driver : BenchmarkDigest
  priv toolkit : BenchmarkDigest
  priv model : BenchmarkDigest
  priv tokenizer : BenchmarkDigest
  priv protocol : BenchmarkDigest
  priv corpus : BenchmarkDigest
  priv randomization : BenchmarkDigest
}

///|
pub fn BenchmarkEnvironment::new(
  hardware~ : BenchmarkDigest,
  driver~ : BenchmarkDigest,
  toolkit~ : BenchmarkDigest,
  model~ : BenchmarkDigest,
  tokenizer~ : BenchmarkDigest,
  protocol~ : BenchmarkDigest,
  corpus~ : BenchmarkDigest,
  randomization~ : BenchmarkDigest,
) -> BenchmarkEnvironment {
  {
    hardware,
    driver,
    toolkit,
    model,
    tokenizer,
    protocol,
    corpus,
    randomization,
  }
}

///|
/// Exact revision and executable/container identity for one engine.
pub struct BenchmarkEngineIdentity {
  priv engine : BenchmarkEngine
  priv revision : BenchmarkDigest
  priv executable : BenchmarkDigest
}

///|
pub fn BenchmarkEngineIdentity::new(
  engine~ : BenchmarkEngine,
  revision~ : BenchmarkDigest,
  executable~ : BenchmarkDigest,
) -> BenchmarkEngineIdentity {
  { engine, revision, executable }
}

///|
pub fn BenchmarkEngineIdentity::engine(
  self : BenchmarkEngineIdentity,
) -> BenchmarkEngine {
  self.engine
}

///|
/// One workload configuration. Entries are supplied in canonical profile
/// order and bind the tokenizer-verified inputs and runner configuration.
pub struct BenchmarkWorkload {
  priv profile : BenchmarkProfile
  priv configuration : BenchmarkDigest
}

///|
pub fn BenchmarkWorkload::new(
  profile~ : BenchmarkProfile,
  configuration~ : BenchmarkDigest,
) -> BenchmarkWorkload {
  { profile, configuration }
}

///|
pub fn BenchmarkWorkload::profile(self : BenchmarkWorkload) -> BenchmarkProfile {
  self.profile
}

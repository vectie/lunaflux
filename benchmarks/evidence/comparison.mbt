///|
/// Canonical evidence admission for one fair three-engine comparison.
pub struct BenchmarkComparison {
  priv digest : BenchmarkDigest
  priv canonical : Bytes
  priv trial_count : Int
}

///|
fn validate_engine_identities(
  engines : ArrayView[BenchmarkEngineIdentity],
) -> Unit raise BenchmarkEvidenceError {
  if engines.length() != 3 ||
    engines[0].engine != LunaFlux ||
    engines[1].engine != Vllm ||
    engines[2].engine != Sglang {
    raise InvalidEngineIdentity
  }
}

///|
fn validate_workloads(
  workloads : ArrayView[BenchmarkWorkload],
) -> Unit raise BenchmarkEvidenceError {
  let profiles = all_profiles()
  if workloads.length() != profiles.length() {
    raise InvalidWorkload
  }
  for index in 0..<profiles.length() {
    if workloads[index].profile != profiles[index] {
      raise InvalidWorkload
    }
  }
}

///|
fn expected_trial_index(
  profile : BenchmarkProfile,
  ordinal : Int,
  engine : BenchmarkEngine,
) -> Int {
  profile_tag(profile) * 9 + (ordinal - 1) * 3 + engine_tag(engine)
}

///|
fn validate_trials(
  trials : ArrayView[BenchmarkTrial],
) -> Unit raise BenchmarkEvidenceError {
  if trials.length() != 81 {
    raise InvalidComparisonShape
  }
  let seen = FixedArray::make(81, false)
  let positions = FixedArray::make(81, false)
  for input_index, trial in trials {
    let index = expected_trial_index(trial.profile, trial.ordinal, trial.engine)
    if seen[index] {
      raise DuplicateTrial
    }
    if index != input_index {
      raise InvalidComparisonShape
    }
    seen[index] = true
    let position_index = profile_tag(trial.profile) * 9 +
      engine_tag(trial.engine) * 3 +
      trial.order_position
    if positions[position_index] {
      raise BiasedEngineOrder
    }
    positions[position_index] = true
  }
  for present in seen {
    if !present {
      raise InvalidComparisonShape
    }
  }
  for present in positions {
    if !present {
      raise BiasedEngineOrder
    }
  }
}

///|
pub fn BenchmarkComparison::admit(
  environment~ : BenchmarkEnvironment,
  engines~ : ArrayView[BenchmarkEngineIdentity],
  workloads~ : ArrayView[BenchmarkWorkload],
  trials~ : ArrayView[BenchmarkTrial],
) -> BenchmarkComparison raise BenchmarkEvidenceError {
  validate_engine_identities(engines)
  validate_workloads(workloads)
  validate_trials(trials)
  let canonical = encode_comparison(environment, engines, workloads, trials)
  let digest = BenchmarkDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
  )
  { digest, canonical, trial_count: trials.length() }
}

///|
pub fn BenchmarkComparison::digest(
  self : BenchmarkComparison,
) -> BenchmarkDigest {
  self.digest
}

///|
pub fn BenchmarkComparison::canonical_bytes(
  self : BenchmarkComparison,
) -> Bytes {
  self.canonical
}

///|
pub fn BenchmarkComparison::trial_count(self : BenchmarkComparison) -> Int {
  self.trial_count
}

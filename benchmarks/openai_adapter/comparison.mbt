///|
pub fn OpenAIComparisonEnvironmentPins::new(
  hardware : @evidence.BenchmarkDigest,
  driver : @evidence.BenchmarkDigest,
  toolkit : @evidence.BenchmarkDigest,
  model : @evidence.BenchmarkDigest,
  randomization : @evidence.BenchmarkDigest,
) -> OpenAIComparisonEnvironmentPins {
  { hardware, driver, toolkit, model, randomization }
}

///|
fn openai_comparison_profiles() -> FixedArray[@evidence.BenchmarkProfile] {
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

///|
fn openai_comparison_engines() -> FixedArray[@evidence.BenchmarkEngine] {
  [LunaFlux, Vllm, Sglang]
}

///|
fn openai_baseline_identities_equal(
  left : OpenAIBaselineIdentity,
  right : OpenAIBaselineIdentity,
) -> Bool {
  left.engine == right.engine &&
  left.revision == right.revision &&
  left.image == right.image &&
  left.configuration == right.configuration
}

///|
fn openai_comparison_shared_inputs_equal(
  left : OpenAIBaselineInputs,
  right : OpenAIBaselineInputs,
) -> Bool {
  left.tokenizer == right.tokenizer &&
  left.corpus == right.corpus &&
  left.protocol == right.protocol
}

///|
fn openai_comparison_expected_position(engine : Int, ordinal : Int) -> Int {
  match ordinal {
    1 => engine
    2 => (engine + 2) % 3
    _ => (engine + 1) % 3
  }
}

///|
fn require_openai_measurement_shape(
  trials : ArrayView[OpenAIBaselineTrialEvidence],
) -> Unit raise OpenAIBaselineAdapterError {
  let profiles = openai_comparison_profiles()
  let engines = openai_comparison_engines()
  for profile_index in 0..<profiles.length() {
    let base = profile_index * 9
    let canonical_inputs = trials[base].inputs
    for ordinal_index in 0..<3 {
      for engine_index in 0..<3 {
        let index = base + ordinal_index * 3 + engine_index
        let trial = trials[index]
        if trial.inputs.profile != profiles[profile_index] ||
          trial.source_identity.engine != engines[engine_index] ||
          trial.ordinal != ordinal_index + 1 ||
          trial.order_position !=
          openai_comparison_expected_position(engine_index, ordinal_index + 1) {
          raise OpenAIBaselineIncompleteTrial
        }
        require_identical_openai_baseline_inputs(canonical_inputs, trial.inputs)
      }
    }
  }
}

///|
fn require_openai_engine_identity(
  trials : ArrayView[OpenAIBaselineTrialEvidence],
  engine_index : Int,
) -> OpenAIBaselineIdentity raise OpenAIBaselineAdapterError {
  let canonical = trials[engine_index].source_identity
  for profile_index in 0..<9 {
    for ordinal_index in 0..<3 {
      let index = profile_index * 9 + ordinal_index * 3 + engine_index
      if !openai_baseline_identities_equal(
          canonical,
          trials[index].source_identity,
        ) {
        raise OpenAIBaselineInputMismatch
      }
    }
  }
  canonical
}

///|
/// Admits only measurements produced by this exact Chat Completions SSE
/// adapter. A native-framed LunaFlux qualification trial has no constructor
/// for this branded set and therefore cannot be mixed with baseline evidence.
/// Independent correctness evidence must be joined later by its owning
/// verifier before calling `BenchmarkComparison::admit`.
pub fn admit_openai_comparison_measurements(
  pins : OpenAIComparisonEnvironmentPins,
  trials : ArrayView[OpenAIBaselineTrialEvidence],
) -> OpenAIComparisonMeasurementSet raise OpenAIBaselineAdapterError {
  if trials.length() != 81 {
    raise OpenAIBaselineIncompleteTrial
  }
  require_openai_measurement_shape(trials)
  let common = trials[0].inputs
  for trial in trials {
    if !openai_comparison_shared_inputs_equal(common, trial.inputs) {
      raise OpenAIBaselineInputMismatch
    }
  }
  let engines = FixedArray::makei(3, engine_index => {
    let identity = require_openai_engine_identity(trials, engine_index)
    @evidence.BenchmarkEngineIdentity::new(
      engine=identity.engine,
      revision=identity.revision,
      executable=openai_baseline_comparison_executable(identity),
    )
  })
  let profiles = openai_comparison_profiles()
  let workloads = FixedArray::makei(9, profile_index => {
    @evidence.BenchmarkWorkload::new(
      profile=profiles[profile_index],
      configuration=trials[profile_index * 9].inputs.workload,
    )
  })
  let owned = FixedArray::makei(81, fn(index) { trials[index] })
  {
    environment: @evidence.BenchmarkEnvironment::new(
      hardware=pins.hardware,
      driver=pins.driver,
      toolkit=pins.toolkit,
      model=pins.model,
      tokenizer=common.tokenizer,
      protocol=trials[0].observed_protocol_digest,
      corpus=common.corpus,
      randomization=pins.randomization,
    ),
    engines,
    workloads,
    trials: owned,
  }
}

///|
/// Refuses promotion of this Chat Completions observation set as Responses
/// evidence. A future Responses observer must implement and brand the actual
/// `/v1/responses` event lifecycle instead of relabeling this set.
pub fn OpenAIComparisonMeasurementSet::require_responses_v1_observation(
  self : OpenAIComparisonMeasurementSet,
) -> Unit raise OpenAIBaselineAdapterError {
  ignore(self)
  raise OpenAIBaselineProtocolIncompatible
}

///|
pub fn OpenAIComparisonMeasurementSet::trial_count(
  self : OpenAIComparisonMeasurementSet,
) -> Int {
  self.trials.length()
}

///|
pub fn OpenAIComparisonMeasurementSet::environment(
  self : OpenAIComparisonMeasurementSet,
) -> @evidence.BenchmarkEnvironment {
  self.environment
}

///|
pub fn OpenAIComparisonMeasurementSet::engine_identity(
  self : OpenAIComparisonMeasurementSet,
  index : Int,
) -> @evidence.BenchmarkEngineIdentity raise OpenAIBaselineAdapterError {
  if index < 0 || index >= self.engines.length() {
    raise OpenAIBaselineInvalidInputs
  }
  self.engines[index]
}

///|
pub fn OpenAIComparisonMeasurementSet::workload(
  self : OpenAIComparisonMeasurementSet,
  index : Int,
) -> @evidence.BenchmarkWorkload raise OpenAIBaselineAdapterError {
  if index < 0 || index >= self.workloads.length() {
    raise OpenAIBaselineInvalidInputs
  }
  self.workloads[index]
}

///|
pub fn OpenAIComparisonMeasurementSet::measurements(
  self : OpenAIComparisonMeasurementSet,
  index : Int,
) -> @evidence.BenchmarkSummaryEvidence raise OpenAIBaselineAdapterError {
  if index < 0 || index >= self.trials.length() {
    raise OpenAIBaselineInvalidInputs
  }
  self.trials[index].measurements
}

///|
fn require_openai_responses_measurement_shape(
  trials : ArrayView[OpenAIResponsesTrialEvidence],
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
          openai_comparison_expected_position(engine_index, ordinal_index + 1) ||
          trial.observed_protocol != ResponsesSseV1 {
          raise OpenAIBaselineIncompleteTrial
        }
        require_identical_openai_baseline_inputs(canonical_inputs, trial.inputs)
      }
    }
  }
}

///|
fn require_openai_responses_engine_identity(
  trials : ArrayView[OpenAIResponsesTrialEvidence],
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
/// Admits an exact three-engine, nine-profile, three-trial Responses
/// measurement matrix. The result remains inert measurement evidence: it has
/// no correctness verifier and no conversion to benchmark approval authority.
pub fn admit_openai_responses_comparison_measurements(
  pins : OpenAIComparisonEnvironmentPins,
  trials : ArrayView[OpenAIResponsesTrialEvidence],
) -> OpenAIResponsesComparisonMeasurementSet raise OpenAIBaselineAdapterError {
  if trials.length() != 81 {
    raise OpenAIBaselineIncompleteTrial
  }
  require_openai_responses_measurement_shape(trials)
  let common = trials[0].inputs
  for trial in trials {
    if !openai_comparison_shared_inputs_equal(common, trial.inputs) {
      raise OpenAIBaselineInputMismatch
    }
  }
  let engines = FixedArray::makei(3, engine_index => {
    let identity = require_openai_responses_engine_identity(
      trials, engine_index,
    )
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
pub fn OpenAIResponsesComparisonMeasurementSet::require_responses_v1_observation(
  self : OpenAIResponsesComparisonMeasurementSet,
) -> Unit {
  ignore(self)
}

///|
pub fn OpenAIResponsesComparisonMeasurementSet::trial_count(
  self : OpenAIResponsesComparisonMeasurementSet,
) -> Int {
  self.trials.length()
}

///|
pub fn OpenAIResponsesComparisonMeasurementSet::environment(
  self : OpenAIResponsesComparisonMeasurementSet,
) -> @evidence.BenchmarkEnvironment {
  self.environment
}

///|
pub fn OpenAIResponsesComparisonMeasurementSet::engine_identity(
  self : OpenAIResponsesComparisonMeasurementSet,
  index : Int,
) -> @evidence.BenchmarkEngineIdentity raise OpenAIBaselineAdapterError {
  if index < 0 || index >= self.engines.length() {
    raise OpenAIBaselineInvalidInputs
  }
  self.engines[index]
}

///|
pub fn OpenAIResponsesComparisonMeasurementSet::workload(
  self : OpenAIResponsesComparisonMeasurementSet,
  index : Int,
) -> @evidence.BenchmarkWorkload raise OpenAIBaselineAdapterError {
  if index < 0 || index >= self.workloads.length() {
    raise OpenAIBaselineInvalidInputs
  }
  self.workloads[index]
}

///|
pub fn OpenAIResponsesComparisonMeasurementSet::measurements(
  self : OpenAIResponsesComparisonMeasurementSet,
  index : Int,
) -> @evidence.BenchmarkSummaryEvidence raise OpenAIBaselineAdapterError {
  if index < 0 || index >= self.trials.length() {
    raise OpenAIBaselineInvalidInputs
  }
  self.trials[index].measurements
}

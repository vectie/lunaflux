///|
fn encode_openai_responses_trial(
  identity : OpenAIBaselineIdentity,
  inputs : OpenAIBaselineInputs,
  measurements : @evidence.BenchmarkSummaryEvidence,
  captured_mode : Bool,
) -> (Bytes, @evidence.BenchmarkDigest) raise OpenAIBaselineAdapterError {
  let comparison_executable = openai_baseline_comparison_executable(identity)
  let observed_protocol = ResponsesSseV1
  let record = "schema=lunaflux-openai-responses-trial.v1\n" +
    "engine=" +
    openai_baseline_engine_name(identity.engine) +
    "\nrevision_sha256=" +
    identity.revision.as_hex() +
    "\nimage_sha256=" +
    identity.image.as_hex() +
    "\nengine_configuration_sha256=" +
    identity.configuration.as_hex() +
    "\ncomparison_executable_sha256=" +
    comparison_executable.as_hex() +
    "\ntokenizer_sha256=" +
    inputs.tokenizer.as_hex() +
    "\ncorpus_sha256=" +
    inputs.corpus.as_hex() +
    "\nobservation_protocol=" +
    openai_baseline_observed_protocol_name(observed_protocol) +
    "\nobservation_protocol_sha256=" +
    openai_baseline_observed_protocol_digest(observed_protocol).as_hex() +
    "\ndeclared_request_protocol_sha256=" +
    inputs.protocol.as_hex() +
    "\nprofile=" +
    openai_baseline_profile_name(inputs.profile) +
    "\nworkload_configuration_sha256=" +
    inputs.workload.as_hex() +
    "\nmeasurement_sha256=" +
    (if captured_mode {
      @runner.captured_monotonic_nanos_measurement_digest().as_hex()
    } else {
      @runner.system_monotonic_millis_measurement_digest().as_hex()
    }) +
    "\nraw_events_sha256=" +
    measurements.raw_events_digest().as_hex() +
    "\nsummary_sha256=" +
    measurements.summary_digest().as_hex() +
    "\ntoken_accounting=terminal_usage.output_tokens.v1\n" +
    "process_lifecycle_authority=external\ncredential_authority=external\n" +
    "comparison_authority=none\ncorrectness_authority=none\n"
  let canonical = @utf8.encode(record)
  let digest = @evidence.BenchmarkDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
  ) catch {
    _ => raise OpenAIBaselineCollectorRejected
  }
  (canonical, digest)
}

///|
pub fn OpenAIResponsesTrialObserver::finish(
  self : OpenAIResponsesTrialObserver,
) -> OpenAIResponsesTrialEvidence raise OpenAIBaselineAdapterError {
  if self.finished || self.submitted != self.lanes.length() {
    raise OpenAIBaselineIncompleteTrial
  }
  for lane in self.lanes {
    if lane.phase != OPENAI_BASELINE_TERMINAL || lane.event_length != 0 {
      raise OpenAIBaselineIncompleteTrial
    }
  }
  let measurements = self.collector.finish() catch {
    _ => raise OpenAIBaselineCollectorRejected
  }
  let (canonical, digest) = encode_openai_responses_trial(
    self.identity,
    self.inputs,
    measurements,
    self.captured_mode,
  )
  let comparison_executable = openai_baseline_comparison_executable(
    self.identity,
  )
  let observed_protocol = ResponsesSseV1
  let observed_protocol_digest = openai_baseline_observed_protocol_digest(
    observed_protocol,
  )
  self.finished = true
  {
    measurements,
    source_identity: self.identity,
    inputs: self.inputs,
    ordinal: self.ordinal,
    order_position: self.order_position,
    identity: @evidence.BenchmarkEngineIdentity::new(
      engine=self.identity.engine,
      revision=self.identity.revision,
      executable=comparison_executable,
    ),
    workload: @evidence.BenchmarkWorkload::new(
      profile=self.inputs.profile,
      configuration=self.inputs.workload,
    ),
    configuration: self.identity.configuration,
    comparison_executable,
    tokenizer: self.inputs.tokenizer,
    corpus: self.inputs.corpus,
    declared_protocol: self.inputs.protocol,
    observed_protocol,
    observed_protocol_digest,
    canonical,
    digest,
  }
}

///|
pub fn OpenAIResponsesTrialEvidence::measurements(
  self : OpenAIResponsesTrialEvidence,
) -> @evidence.BenchmarkSummaryEvidence {
  self.measurements
}

///|
pub fn OpenAIResponsesTrialEvidence::engine_identity(
  self : OpenAIResponsesTrialEvidence,
) -> @evidence.BenchmarkEngineIdentity {
  self.identity
}

///|
pub fn OpenAIResponsesTrialEvidence::workload(
  self : OpenAIResponsesTrialEvidence,
) -> @evidence.BenchmarkWorkload {
  self.workload
}

///|
pub fn OpenAIResponsesTrialEvidence::configuration_digest(
  self : OpenAIResponsesTrialEvidence,
) -> @evidence.BenchmarkDigest {
  self.configuration
}

///|
pub fn OpenAIResponsesTrialEvidence::revision_digest(
  self : OpenAIResponsesTrialEvidence,
) -> @evidence.BenchmarkDigest {
  self.source_identity.revision
}

///|
pub fn OpenAIResponsesTrialEvidence::image_digest(
  self : OpenAIResponsesTrialEvidence,
) -> @evidence.BenchmarkDigest {
  self.source_identity.image
}

///|
pub fn OpenAIResponsesTrialEvidence::comparison_executable_digest(
  self : OpenAIResponsesTrialEvidence,
) -> @evidence.BenchmarkDigest {
  self.comparison_executable
}

///|
pub fn OpenAIResponsesTrialEvidence::tokenizer_digest(
  self : OpenAIResponsesTrialEvidence,
) -> @evidence.BenchmarkDigest {
  self.tokenizer
}

///|
pub fn OpenAIResponsesTrialEvidence::corpus_digest(
  self : OpenAIResponsesTrialEvidence,
) -> @evidence.BenchmarkDigest {
  self.corpus
}

///|
pub fn OpenAIResponsesTrialEvidence::protocol_digest(
  self : OpenAIResponsesTrialEvidence,
) -> @evidence.BenchmarkDigest {
  self.observed_protocol_digest
}

///|
pub fn OpenAIResponsesTrialEvidence::observed_protocol(
  self : OpenAIResponsesTrialEvidence,
) -> OpenAIBaselineObservedProtocol {
  self.observed_protocol
}

///|
pub fn OpenAIResponsesTrialEvidence::declared_request_protocol_digest(
  self : OpenAIResponsesTrialEvidence,
) -> @evidence.BenchmarkDigest {
  self.declared_protocol
}

///|
pub fn OpenAIResponsesTrialEvidence::canonical_bytes(
  self : OpenAIResponsesTrialEvidence,
) -> Bytes {
  self.canonical
}

///|
pub fn OpenAIResponsesTrialEvidence::digest(
  self : OpenAIResponsesTrialEvidence,
) -> @evidence.BenchmarkDigest {
  self.digest
}

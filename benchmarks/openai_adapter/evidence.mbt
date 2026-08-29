///|
fn openai_baseline_engine_name(engine : @evidence.BenchmarkEngine) -> String {
  match engine {
    Vllm => "vllm"
    Sglang => "sglang"
    LunaFlux => "lunaflux"
  }
}

///|
fn openai_baseline_profile_name(profile : @evidence.BenchmarkProfile) -> String {
  match profile {
    Latency => "latency"
    Chat => "chat"
    LongPrefill => "long-prefill"
    DecodeHeavy => "decode-heavy"
    PrefixRich => "prefix-rich"
    PrefixCold => "prefix-cold"
    Saturation => "saturation"
    Churn => "churn"
    Mixed => "mixed"
  }
}

///|
fn openai_baseline_observed_protocol_name(
  protocol : OpenAIBaselineObservedProtocol,
) -> String {
  match protocol {
    ChatCompletionsSseV1 => "openai.chat.completions.sse.v1"
    ResponsesSseV1 => "openai.responses.sse.v1"
  }
}

///|
fn openai_baseline_observed_protocol_digest(
  protocol : OpenAIBaselineObservedProtocol,
) -> @evidence.BenchmarkDigest raise OpenAIBaselineAdapterError {
  let canonical = "schema=lunaflux-openai-observed-protocol.v1\nprotocol=" +
    openai_baseline_observed_protocol_name(protocol) +
    "\n"
  @evidence.BenchmarkDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(@utf8.encode(canonical))),
  ) catch {
    _ => raise OpenAIBaselineCollectorRejected
  }
}

///|
fn openai_baseline_comparison_executable(
  identity : OpenAIBaselineIdentity,
) -> @evidence.BenchmarkDigest raise OpenAIBaselineAdapterError {
  let canonical = "schema=lunaflux-openai-baseline-container-identity.v1\n" +
    "engine=" +
    openai_baseline_engine_name(identity.engine) +
    "\nimage_sha256=" +
    identity.image.as_hex() +
    "\nengine_configuration_sha256=" +
    identity.configuration.as_hex() +
    "\n"
  @evidence.BenchmarkDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(@utf8.encode(canonical))),
  ) catch {
    _ => raise OpenAIBaselineCollectorRejected
  }
}

///|
fn encode_openai_baseline_trial(
  identity : OpenAIBaselineIdentity,
  inputs : OpenAIBaselineInputs,
  measurements : @evidence.BenchmarkSummaryEvidence,
) -> (Bytes, @evidence.BenchmarkDigest) raise OpenAIBaselineAdapterError {
  let comparison_executable = openai_baseline_comparison_executable(identity)
  let observed_protocol = ChatCompletionsSseV1
  let record = "schema=lunaflux-openai-baseline-trial.v1\n" +
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
    @runner.system_monotonic_millis_measurement_digest().as_hex() +
    "\nraw_events_sha256=" +
    measurements.raw_events_digest().as_hex() +
    "\nsummary_sha256=" +
    measurements.summary_digest().as_hex() +
    "\nprocess_lifecycle_authority=external\ncredential_authority=external\n"
  let canonical = @utf8.encode(record)
  let digest = @evidence.BenchmarkDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
  ) catch {
    _ => raise OpenAIBaselineCollectorRejected
  }
  (canonical, digest)
}

///|
pub fn OpenAIBaselineTrialAdapter::finish(
  self : OpenAIBaselineTrialAdapter,
) -> OpenAIBaselineTrialEvidence raise OpenAIBaselineAdapterError {
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
  let (canonical, digest) = encode_openai_baseline_trial(
    self.identity,
    self.inputs,
    measurements,
  )
  let comparison_executable = openai_baseline_comparison_executable(
    self.identity,
  )
  let observed_protocol = ChatCompletionsSseV1
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
pub fn OpenAIBaselineTrialEvidence::measurements(
  self : OpenAIBaselineTrialEvidence,
) -> @evidence.BenchmarkSummaryEvidence {
  self.measurements
}

///|
pub fn OpenAIBaselineTrialEvidence::engine_identity(
  self : OpenAIBaselineTrialEvidence,
) -> @evidence.BenchmarkEngineIdentity {
  self.identity
}

///|
pub fn OpenAIBaselineTrialEvidence::workload(
  self : OpenAIBaselineTrialEvidence,
) -> @evidence.BenchmarkWorkload {
  self.workload
}

///|
pub fn OpenAIBaselineTrialEvidence::configuration_digest(
  self : OpenAIBaselineTrialEvidence,
) -> @evidence.BenchmarkDigest {
  self.configuration
}

///|
pub fn OpenAIBaselineTrialEvidence::revision_digest(
  self : OpenAIBaselineTrialEvidence,
) -> @evidence.BenchmarkDigest {
  self.source_identity.revision
}

///|
pub fn OpenAIBaselineTrialEvidence::image_digest(
  self : OpenAIBaselineTrialEvidence,
) -> @evidence.BenchmarkDigest {
  self.source_identity.image
}

///|
pub fn OpenAIBaselineTrialEvidence::comparison_executable_digest(
  self : OpenAIBaselineTrialEvidence,
) -> @evidence.BenchmarkDigest {
  self.comparison_executable
}

///|
pub fn OpenAIBaselineTrialEvidence::tokenizer_digest(
  self : OpenAIBaselineTrialEvidence,
) -> @evidence.BenchmarkDigest {
  self.tokenizer
}

///|
pub fn OpenAIBaselineTrialEvidence::corpus_digest(
  self : OpenAIBaselineTrialEvidence,
) -> @evidence.BenchmarkDigest {
  self.corpus
}

///|
pub fn OpenAIBaselineTrialEvidence::protocol_digest(
  self : OpenAIBaselineTrialEvidence,
) -> @evidence.BenchmarkDigest {
  self.observed_protocol_digest
}

///|
pub fn OpenAIBaselineTrialEvidence::observed_protocol(
  self : OpenAIBaselineTrialEvidence,
) -> OpenAIBaselineObservedProtocol {
  self.observed_protocol
}

///|
pub fn OpenAIBaselineTrialEvidence::declared_request_protocol_digest(
  self : OpenAIBaselineTrialEvidence,
) -> @evidence.BenchmarkDigest {
  self.declared_protocol
}

///|
pub fn OpenAIBaselineTrialEvidence::canonical_bytes(
  self : OpenAIBaselineTrialEvidence,
) -> Bytes {
  self.canonical
}

///|
pub fn OpenAIBaselineTrialEvidence::digest(
  self : OpenAIBaselineTrialEvidence,
) -> @evidence.BenchmarkDigest {
  self.digest
}

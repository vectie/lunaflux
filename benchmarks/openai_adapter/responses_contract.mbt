///|
/// Creates one shared Responses observation contract. Engine provenance stays
/// bound to the caller's exact revision, image, and configuration digests.
pub fn OpenAIResponsesTrialContract::new(
  identity : OpenAIBaselineIdentity,
  inputs : OpenAIBaselineInputs,
  ordinal : Int,
  order_position : Int,
  maximum_event_bytes : Int,
  maximum_json_depth : Int,
) -> OpenAIResponsesTrialContract raise OpenAIBaselineAdapterError {
  if ordinal < 1 || ordinal > 3 || order_position < 0 || order_position > 2 {
    raise OpenAIBaselineInvalidInputs
  }
  if maximum_event_bytes < 64 ||
    maximum_event_bytes > 1048576 ||
    maximum_json_depth <= 0 ||
    maximum_json_depth > 32 {
    raise OpenAIBaselineInvalidLimits
  }
  {
    identity,
    inputs,
    ordinal,
    order_position,
    maximum_event_bytes,
    maximum_json_depth,
  }
}

///|
fn new_openai_responses_lane(capacity : Int) -> OpenAIResponsesLane {
  {
    event_storage: FixedArray::make(capacity, b'\x00'),
    event_length: 0,
    phase: OPENAI_BASELINE_VACANT,
    response_id: None,
    created_seen: false,
    first_delta_seen: false,
    observed_delta_events: 0L,
    usage_seen: false,
    completion_tokens: 0L,
    terminal_kind: OPENAI_RESPONSES_NO_TERMINAL,
  }
}

///|
pub fn OpenAIResponsesTrialContract::start(
  self : OpenAIResponsesTrialContract,
) -> OpenAIResponsesTrialObserver raise OpenAIBaselineAdapterError {
  let capacity = self.inputs.request_input_tokens.length()
  let collector = @runner.BenchmarkTrialCollector::new(
    engine=self.identity.engine,
    profile=self.inputs.profile,
    ordinal=self.ordinal,
    order_position=self.order_position,
    request_capacity=capacity,
  ) catch {
    _ => raise OpenAIBaselineCollectorRejected
  }
  {
    identity: self.identity,
    inputs: self.inputs,
    collector,
    lanes: FixedArray::makei(capacity, fn(_) {
      new_openai_responses_lane(self.maximum_event_bytes)
    }),
    maximum_json_depth: self.maximum_json_depth,
    ordinal: self.ordinal,
    order_position: self.order_position,
    submitted: 0,
    finished: false,
    captured_mode: false,
    captured_timestamp: None,
  }
}

///|
/// Starts an offline observer that only accepts explicitly captured monotonic
/// timestamps. The resulting evidence uses the captured-clock measurement
/// identity and cannot be confused with this process's live clock.
pub fn OpenAIResponsesTrialContract::start_captured(
  self : OpenAIResponsesTrialContract,
) -> OpenAIResponsesTrialObserver raise OpenAIBaselineAdapterError {
  let capacity = self.inputs.request_input_tokens.length()
  let collector = @runner.BenchmarkTrialCollector::new_captured(
    engine=self.identity.engine,
    profile=self.inputs.profile,
    ordinal=self.ordinal,
    order_position=self.order_position,
    request_capacity=capacity,
  ) catch {
    _ => raise OpenAIBaselineCollectorRejected
  }
  {
    identity: self.identity,
    inputs: self.inputs,
    collector,
    lanes: FixedArray::makei(capacity, fn(_) {
      new_openai_responses_lane(self.maximum_event_bytes)
    }),
    maximum_json_depth: self.maximum_json_depth,
    ordinal: self.ordinal,
    order_position: self.order_position,
    submitted: 0,
    finished: false,
    captured_mode: true,
    captured_timestamp: None,
  }
}

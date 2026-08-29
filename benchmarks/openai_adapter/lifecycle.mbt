///|
fn new_openai_baseline_lane(capacity : Int) -> OpenAIBaselineLane {
  {
    event_storage: FixedArray::make(capacity, b'\x00'),
    event_length: 0,
    phase: OPENAI_BASELINE_VACANT,
    first_content_seen: false,
    observed_tokens: 0L,
    finish_seen: false,
    usage_seen: false,
    completion_tokens: 0L,
  }
}

///|
pub fn OpenAIBaselineTrialContract::start(
  self : OpenAIBaselineTrialContract,
) -> OpenAIBaselineTrialAdapter raise OpenAIBaselineAdapterError {
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
      new_openai_baseline_lane(self.maximum_event_bytes)
    }),
    maximum_json_depth: self.maximum_json_depth,
    ordinal: self.ordinal,
    order_position: self.order_position,
    submitted: 0,
    finished: false,
  }
}

///|
fn OpenAIBaselineTrialAdapter::require_open_lane(
  self : OpenAIBaselineTrialAdapter,
  request_ordinal : Int,
) -> OpenAIBaselineLane raise OpenAIBaselineAdapterError {
  if self.finished ||
    request_ordinal < 0 ||
    request_ordinal >= self.lanes.length() {
    raise OpenAIBaselineInvalidLifecycle
  }
  self.lanes[request_ordinal]
}

///|
/// Samples submission immediately before the external transport owner writes
/// this request. The token count comes only from the immutable shared input
/// declaration and cannot differ between baseline engines.
pub fn OpenAIBaselineTrialAdapter::submit(
  self : OpenAIBaselineTrialAdapter,
  request_ordinal : Int,
) -> Unit raise OpenAIBaselineAdapterError {
  let lane = self.require_open_lane(request_ordinal)
  if request_ordinal != self.submitted || lane.phase != OPENAI_BASELINE_VACANT {
    raise OpenAIBaselineInvalidLifecycle
  }
  self.collector.submit(
    request_ordinal~,
    input_tokens=self.inputs.request_input_tokens[request_ordinal],
  ) catch {
    _ => raise OpenAIBaselineCollectorRejected
  }
  lane.phase = OPENAI_BASELINE_SUBMITTED
  self.submitted += 1
}

///|
fn openai_event_stream_content_type(value : StringView) -> Bool {
  value == "text/event-stream" || value == "text/event-stream; charset=utf-8"
}

///|
/// Maps only an exact HTTP 200 event-stream head to admission. Explicit
/// non-success statuses terminate without parsing a body; unclassified status
/// or media-type combinations are rejected as ambiguous.
pub fn OpenAIBaselineTrialAdapter::observe_response_head(
  self : OpenAIBaselineTrialAdapter,
  request_ordinal : Int,
  status : Int,
  content_type : StringView,
) -> Unit raise OpenAIBaselineAdapterError {
  let lane = self.require_open_lane(request_ordinal)
  if lane.phase != OPENAI_BASELINE_SUBMITTED {
    raise OpenAIBaselineInvalidLifecycle
  }
  if status == 200 {
    if !openai_event_stream_content_type(content_type) {
      raise OpenAIBaselineInvalidResponseHead
    }
    self.collector.admit(request_ordinal~) catch {
      _ => raise OpenAIBaselineCollectorRejected
    }
    lane.phase = OPENAI_BASELINE_ADMITTED
    return
  }
  if content_type != "application/json" {
    raise OpenAIBaselineInvalidResponseHead
  }
  let outcome = match status {
    400 | 401 | 403 | 404 | 409 | 422 | 429 => @evidence.Rejected
    408 | 504 => TimedOut
    500 | 502 | 503 => Failed
    _ => raise OpenAIBaselineAmbiguousEvent
  }
  self.collector.terminate(request_ordinal~, outcome~, generated_tokens=0L) catch {
    _ => raise OpenAIBaselineCollectorRejected
  }
  lane.phase = OPENAI_BASELINE_TERMINAL
}

///|
/// Records transport termination only for outcomes that have no ambiguous
/// OpenAI semantic mapping. Content-bearing streams use their required token
/// logprobs for exact generated-token accounting on transport termination.
pub fn OpenAIBaselineTrialAdapter::terminate_transport(
  self : OpenAIBaselineTrialAdapter,
  request_ordinal : Int,
  outcome : @evidence.BenchmarkRequestOutcome,
) -> Unit raise OpenAIBaselineAdapterError {
  let lane = self.require_open_lane(request_ordinal)
  if (
      lane.phase != OPENAI_BASELINE_SUBMITTED &&
      lane.phase != OPENAI_BASELINE_ADMITTED
    ) ||
    (outcome != TimedOut && outcome != Cancelled && outcome != Failed) {
    raise OpenAIBaselineAmbiguousEvent
  }
  self.collector.terminate(
    request_ordinal~,
    outcome~,
    generated_tokens=lane.observed_tokens,
  ) catch {
    _ => raise OpenAIBaselineCollectorRejected
  }
  lane.phase = OPENAI_BASELINE_TERMINAL
}

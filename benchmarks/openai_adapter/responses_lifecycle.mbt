///|
fn OpenAIResponsesTrialObserver::require_open_lane(
  self : OpenAIResponsesTrialObserver,
  request_ordinal : Int,
) -> OpenAIResponsesLane raise OpenAIBaselineAdapterError {
  if self.finished ||
    request_ordinal < 0 ||
    request_ordinal >= self.lanes.length() {
    raise OpenAIBaselineInvalidLifecycle
  }
  self.lanes[request_ordinal]
}

///|
pub fn OpenAIResponsesTrialObserver::submit(
  self : OpenAIResponsesTrialObserver,
  request_ordinal : Int,
) -> Unit raise OpenAIBaselineAdapterError {
  if self.captured_mode {
    raise OpenAIBaselineInvalidLifecycle
  }
  self.submit_observed(request_ordinal)
}

///|
fn OpenAIResponsesTrialObserver::submit_observed(
  self : OpenAIResponsesTrialObserver,
  request_ordinal : Int,
) -> Unit raise OpenAIBaselineAdapterError {
  let lane = self.require_open_lane(request_ordinal)
  if request_ordinal != self.submitted || lane.phase != OPENAI_BASELINE_VACANT {
    raise OpenAIBaselineInvalidLifecycle
  }
  let input_tokens = self.inputs.request_input_tokens[request_ordinal]
  self.collector_submit_observed(request_ordinal, input_tokens)
  lane.phase = OPENAI_BASELINE_SUBMITTED
  self.submitted += 1
}

///|
pub fn OpenAIResponsesTrialObserver::submit_captured(
  self : OpenAIResponsesTrialObserver,
  request_ordinal : Int,
  timestamp : @runner.CapturedMonotonicTimestamp,
) -> Unit raise OpenAIBaselineAdapterError {
  if !self.captured_mode {
    raise OpenAIBaselineInvalidLifecycle
  }
  self.captured_timestamp = Some(timestamp)
  self.submit_observed(request_ordinal)
  self.captured_timestamp = None
}

///|
pub fn OpenAIResponsesTrialObserver::observe_response_head(
  self : OpenAIResponsesTrialObserver,
  request_ordinal : Int,
  status : Int,
  content_type : StringView,
) -> Unit raise OpenAIBaselineAdapterError {
  if self.captured_mode {
    raise OpenAIBaselineInvalidLifecycle
  }
  self.observe_response_head_observed(request_ordinal, status, content_type)
}

///|
fn OpenAIResponsesTrialObserver::observe_response_head_observed(
  self : OpenAIResponsesTrialObserver,
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
    self.collector_admit_observed(request_ordinal)
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
  self.collector_terminate_observed(request_ordinal, outcome, 0L)
  lane.phase = OPENAI_BASELINE_TERMINAL
}

///|
pub fn OpenAIResponsesTrialObserver::observe_response_head_captured(
  self : OpenAIResponsesTrialObserver,
  request_ordinal : Int,
  status : Int,
  content_type : StringView,
  timestamp : @runner.CapturedMonotonicTimestamp,
) -> Unit raise OpenAIBaselineAdapterError {
  if !self.captured_mode {
    raise OpenAIBaselineInvalidLifecycle
  }
  self.captured_timestamp = Some(timestamp)
  self.observe_response_head_observed(request_ordinal, status, content_type)
  self.captured_timestamp = None
}

///|
pub fn OpenAIResponsesTrialObserver::terminate_transport(
  self : OpenAIResponsesTrialObserver,
  request_ordinal : Int,
  outcome : @evidence.BenchmarkRequestOutcome,
) -> Unit raise OpenAIBaselineAdapterError {
  if self.captured_mode {
    raise OpenAIBaselineInvalidLifecycle
  }
  self.terminate_transport_observed(request_ordinal, outcome)
}

///|
fn OpenAIResponsesTrialObserver::terminate_transport_observed(
  self : OpenAIResponsesTrialObserver,
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
  if lane.observed_delta_events != 0L {
    raise OpenAIBaselineAmbiguousEvent
  }
  self.collector_terminate_observed(request_ordinal, outcome, 0L)
  lane.phase = OPENAI_BASELINE_TERMINAL
}

///|
pub fn OpenAIResponsesTrialObserver::terminate_transport_captured(
  self : OpenAIResponsesTrialObserver,
  request_ordinal : Int,
  outcome : @evidence.BenchmarkRequestOutcome,
  timestamp : @runner.CapturedMonotonicTimestamp,
) -> Unit raise OpenAIBaselineAdapterError {
  if !self.captured_mode {
    raise OpenAIBaselineInvalidLifecycle
  }
  self.captured_timestamp = Some(timestamp)
  self.terminate_transport_observed(request_ordinal, outcome)
  self.captured_timestamp = None
}

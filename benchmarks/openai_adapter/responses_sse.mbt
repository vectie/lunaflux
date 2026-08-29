///|
fn openai_responses_decode_event(
  storage : FixedArray[Byte],
  length : Int,
) -> String raise OpenAIBaselineAdapterError {
  let bytes = Bytes::makei(length, fn(index) { storage[index] })
  @utf8.decode(bytes) catch {
    _ => raise OpenAIBaselineInvalidSse
  }
}

///|
fn OpenAIResponsesTrialObserver::observe_sse_event(
  self : OpenAIResponsesTrialObserver,
  request_ordinal : Int,
  event_length : Int,
) -> Unit raise OpenAIBaselineAdapterError {
  let lane = self.lanes[request_ordinal]
  let event = openai_responses_decode_event(lane.event_storage, event_length)
  let (payload, declared_event) = if event.has_prefix("data: ") {
    if event.contains("\n") || event.contains("\r") {
      raise OpenAIBaselineInvalidSse
    }
    (event[6:], None)
  } else if event.has_prefix("event: ") {
    let (header, data) = match event.split_once("\r\n") {
      Some(parts) => parts
      None =>
        match event.split_once("\n") {
          Some(parts) => parts
          None => raise OpenAIBaselineInvalidSse
        }
    }
    if !data.has_prefix("data: ") || data.contains("\n") || data.contains("\r") {
      raise OpenAIBaselineInvalidSse
    }
    let declared = header[7:].to_owned()
    if declared.is_empty() {
      raise OpenAIBaselineInvalidSse
    }
    (data[6:], Some(declared))
  } else {
    raise OpenAIBaselineInvalidSse
  }
  if payload == "[DONE]" {
    match declared_event {
      None | Some("done") => ()
      _ => raise OpenAIBaselineAmbiguousEvent
    }
    if lane.terminal_kind == OPENAI_RESPONSES_NO_TERMINAL {
      raise OpenAIBaselineAmbiguousEvent
    }
    let (outcome, generated_tokens) = match lane.terminal_kind {
      OPENAI_RESPONSES_COMPLETED =>
        if lane.usage_seen {
          (@evidence.Completed, lane.completion_tokens)
        } else {
          raise OpenAIBaselineAmbiguousEvent
        }
      OPENAI_RESPONSES_CANCELLED =>
        if lane.usage_seen {
          (Cancelled, lane.completion_tokens)
        } else {
          raise OpenAIBaselineAmbiguousEvent
        }
      OPENAI_RESPONSES_FAILED =>
        if lane.usage_seen {
          (Failed, lane.completion_tokens)
        } else if lane.observed_delta_events == 0L {
          (Failed, 0L)
        } else {
          raise OpenAIBaselineAmbiguousEvent
        }
      _ => raise OpenAIBaselineAmbiguousEvent
    }
    self.collector_terminate_observed(
      request_ordinal, outcome, generated_tokens,
    )
    lane.phase = OPENAI_BASELINE_TERMINAL
    return
  }
  if lane.terminal_kind != OPENAI_RESPONSES_NO_TERMINAL {
    raise OpenAIBaselineAmbiguousEvent
  }
  self.observe_json_event(request_ordinal, payload, declared_event)
}

///|
/// Consumes bounded, arbitrarily split Responses SSE bytes. The semantic
/// parser accepts only one-data-line events and exact measured semantics.
pub fn OpenAIResponsesTrialObserver::offer_sse_bytes(
  self : OpenAIResponsesTrialObserver,
  request_ordinal : Int,
  source : BytesView,
) -> Unit raise OpenAIBaselineAdapterError {
  if self.captured_mode {
    raise OpenAIBaselineInvalidLifecycle
  }
  self.offer_sse_bytes_observed(request_ordinal, source)
}

///|
fn OpenAIResponsesTrialObserver::offer_sse_bytes_observed(
  self : OpenAIResponsesTrialObserver,
  request_ordinal : Int,
  source : BytesView,
) -> Unit raise OpenAIBaselineAdapterError {
  let lane = self.require_open_lane(request_ordinal)
  if lane.phase != OPENAI_BASELINE_ADMITTED || source.length() <= 0 {
    raise OpenAIBaselineInvalidLifecycle
  }
  for byte in source {
    if lane.event_length >= lane.event_storage.length() {
      raise OpenAIBaselineEventTooLarge
    }
    lane.event_storage[lane.event_length] = byte
    lane.event_length += 1
    let delimiter = openai_baseline_event_delimiter_length(
      lane.event_storage,
      lane.event_length,
    )
    if delimiter > 0 {
      let event_length = lane.event_length - delimiter
      if event_length <= 0 {
        raise OpenAIBaselineInvalidSse
      }
      self.observe_sse_event(request_ordinal, event_length)
      lane.event_length = 0
    }
  }
}

///|
pub fn OpenAIResponsesTrialObserver::offer_sse_bytes_captured(
  self : OpenAIResponsesTrialObserver,
  request_ordinal : Int,
  source : BytesView,
  timestamp : @runner.CapturedMonotonicTimestamp,
) -> Unit raise OpenAIBaselineAdapterError {
  if !self.captured_mode {
    raise OpenAIBaselineInvalidLifecycle
  }
  self.captured_timestamp = Some(timestamp)
  self.offer_sse_bytes_observed(request_ordinal, source)
  self.captured_timestamp = None
}

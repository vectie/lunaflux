///|
fn openai_baseline_event_delimiter_length(
  storage : FixedArray[Byte],
  length : Int,
) -> Int {
  if length >= 2 && storage[length - 2] == b'\n' && storage[length - 1] == b'\n' {
    2
  } else if length >= 4 &&
    storage[length - 4] == b'\r' &&
    storage[length - 3] == b'\n' &&
    storage[length - 2] == b'\r' &&
    storage[length - 1] == b'\n' {
    4
  } else {
    0
  }
}

///|
fn openai_baseline_decode_event(
  storage : FixedArray[Byte],
  length : Int,
) -> String raise OpenAIBaselineAdapterError {
  let bytes = Bytes::makei(length, fn(index) { storage[index] })
  @utf8.decode(bytes) catch {
    _ => raise OpenAIBaselineInvalidSse
  }
}

///|
fn OpenAIBaselineTrialAdapter::observe_sse_event(
  self : OpenAIBaselineTrialAdapter,
  request_ordinal : Int,
  event_length : Int,
) -> Unit raise OpenAIBaselineAdapterError {
  let lane = self.lanes[request_ordinal]
  let event = openai_baseline_decode_event(lane.event_storage, event_length)
  if !event.has_prefix("data: ") ||
    event.contains("\ndata:") ||
    event.contains("\r\ndata:") {
    raise OpenAIBaselineInvalidSse
  }
  let payload = event[6:]
  if payload == "[DONE]" {
    if !lane.finish_seen || !lane.usage_seen {
      raise OpenAIBaselineAmbiguousEvent
    }
    self.collector.terminate(
      request_ordinal~,
      outcome=Completed,
      generated_tokens=lane.completion_tokens,
    ) catch {
      _ => raise OpenAIBaselineCollectorRejected
    }
    lane.phase = OPENAI_BASELINE_TERMINAL
    return
  }
  if lane.finish_seen && lane.usage_seen {
    raise OpenAIBaselineAmbiguousEvent
  }
  self.observe_json_event(request_ordinal, payload)
}

///|
/// Consumes arbitrarily split SSE bytes into the startup-fixed lane buffer.
/// Only one `data:` line per event is accepted. Both LF and CRLF blank-line
/// delimiters are recognized; mixed or bare carriage returns fail closed.
pub fn OpenAIBaselineTrialAdapter::offer_sse_bytes(
  self : OpenAIBaselineTrialAdapter,
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

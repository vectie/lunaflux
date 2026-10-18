///|
/// Presentation-only accumulation. Native inference continues to deliver the
/// same token/usage/terminal events; no scheduler or model policy is changed.
/// Consume at most one byte per work unit and retain only bounded visible text.
fn LunaOpenAICompatWorkspace::progress_buffered_payload(
  self : Self,
  event : @luna_event.LunaEventView,
) -> Unit raise LunaOpenAICompatError {
  if self.source_cursor < self.payload_length {
    guard self.buffered_length < self.buffered_text.length() else {
      raise LunaOpenAICompatInvalid(LunaOpenAICompatCapacity)
    }
    self.buffered_text[self.buffered_length] = self.payload_byte_at(
      event,
      self.source_cursor,
    )
    self.buffered_length += 1
    self.source_cursor += 1
    return
  }
  self.source_cursor = 0
  if self.event_kind == LUNA_OPENAI_EVENT_COMPLETED {
    self.payload_length = self.buffered_length
    self.phase = LUNA_OPENAI_COMPAT_CONFIGURE_PART
  } else {
    self.phase = LUNA_OPENAI_COMPAT_FINALIZING
  }
}

///|
fn luna_openai_buffered_literal(
  mode : Int,
  event_kind : Int,
  part : Int,
) -> Bytes {
  if mode == 1 {
    match (event_kind, part) {
      (LUNA_OPENAI_EVENT_COMPLETED, 0) => b"{\"id\":\""
      (LUNA_OPENAI_EVENT_COMPLETED, 7) =>
        b"\",\"output\":[{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\""
      (LUNA_OPENAI_EVENT_COMPLETED, 9) => b"\"}]}],\"usage\":{\"input_tokens\":"
      (LUNA_OPENAI_EVENT_COMPLETED, 17) => b"}}"
      (LUNA_OPENAI_EVENT_FAILED, 0) => b"{\"id\":\""
      (LUNA_OPENAI_EVENT_FAILED, 7) => b"}}"
      _ => luna_openai_responses_literal(event_kind, part)
    }
  } else {
    match (event_kind, part) {
      (LUNA_OPENAI_EVENT_COMPLETED, 0) | (LUNA_OPENAI_EVENT_FAILED, 0) =>
        b"{\"id\":\""
      (LUNA_OPENAI_EVENT_COMPLETED, 3) | (LUNA_OPENAI_EVENT_FAILED, 3) =>
        b"\",\"object\":\"chat.completion\",\"model\":\""
      (LUNA_OPENAI_EVENT_COMPLETED, 5) =>
        b"\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\""
      (LUNA_OPENAI_EVENT_COMPLETED, 17) => b"}}"
      (LUNA_OPENAI_EVENT_FAILED, 9) => b",\"type\":\"lunaflux_error\"}}"
      _ => luna_openai_chat_literal(event_kind, part)
    }
  }
}

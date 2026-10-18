///|
/// Immutable presentation protocol selected by the model adapter. A prompt
/// that opens a non-answer section can declare its closing delimiter here.
/// This does not alter generated tokens, stop decisions, sampling or usage.
pub struct LunaOpenAIOutputFraming {
  priv delimiter : Bytes
  priv fallback : FixedArray[Int]
}

///|
pub fn LunaOpenAIOutputFraming::answer_after(
  delimiter : Bytes,
) -> Self raise LunaOpenAICompatError {
  guard delimiter.length() > 0 && delimiter.length() <= 64 else {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatTemplate)
  }
  for byte in delimiter {
    guard byte >= b'\x20' && byte <= b'\x7e' else {
      raise LunaOpenAICompatInvalid(LunaOpenAICompatTemplate)
    }
  }
  let fallback = FixedArray::make(delimiter.length(), 0)
  let mut matched = 0
  for i in 1..<delimiter.length() {
    while matched > 0 && delimiter[i] != delimiter[matched] {
      matched = fallback[matched - 1]
    }
    if delimiter[i] == delimiter[matched] {
      matched += 1
    }
    fallback[i] = matched
  }
  { delimiter, fallback, }
}

///|
fn LunaOpenAIOutputFraming::advance(
  self : Self,
  matched : Int,
  byte : Byte,
) -> Int {
  let mut next = matched
  while next > 0 && byte != self.delimiter[next] {
    next = self.fallback[next - 1]
  }
  if byte == self.delimiter[next] {
    next += 1
  }
  next
}

///|
/// One input byte per renderer work unit. The delimiter may span any number
/// of token events. Only a scalar match state survives between events; after
/// the first delimiter all following answer text is preserved literally.
fn LunaOpenAICompatWorkspace::progress_output_framing(
  self : Self,
  event : @luna_event.LunaEventView,
) -> Unit raise LunaOpenAICompatError {
  guard self.output_framing is Some(framing) else {
    abort("output framing phase invariant")
  }
  if self.source_cursor < self.payload_length {
    self.framing_matched = framing.advance(
      self.framing_matched,
      self.payload_byte_at(event, self.source_cursor),
    )
    self.source_cursor += 1
    if self.framing_matched != framing.delimiter.length() {
      return
    }
    self.answer_started = true
  }
  self.payload_offset = self.source_cursor
  self.payload_length -= self.payload_offset
  self.source_cursor = 0
  if self.event_kind == LUNA_OPENAI_EVENT_TOKEN && self.payload_length == 0 {
    self.phase = LUNA_OPENAI_COMPAT_FINALIZING
  } else {
    self.finish_visible_setup()
  }
}

///|
pub fn IncrementalOutput::new(
  tokenizer : @tokenizer.TokenizerSpec,
  stops : @inference.StopConditions,
  inference : @inference.InferenceLimits,
  special_policy~ : @tokenizer.DecodeSpecialPolicy,
) -> IncrementalOutput raise IncrementalOutputError {
  let patterns = build_patterns(stops)
  let longest = longest_pattern(patterns)
  let token_bytes = tokenizer.limits().max_token_bytes
  let delta_bytes = inference.max_decoded_delta_bytes()
  if token_bytes > delta_bytes || longest > delta_bytes - token_bytes - 3 {
    raise invalid(DecodeEnvelope, Unsupported)
  }
  {
    tokenizer,
    special_policy,
    patterns,
    states: FixedArray::make(patterns.length(), 0),
    next_states: FixedArray::make(patterns.length(), 0),
    pending: FixedArray::make(longest + 3, b'\x00'),
    scratch: FixedArray::make(delta_bytes, b'\x00'),
    pending_count: 0,
    utf8_remaining: 0,
    utf8_codepoint: 0,
    utf8_minimum: 0,
    utf8_seen: 0,
    lifecycle: OUTPUT_ACTIVE,
  }
}

///|
pub fn IncrementalOutput::is_stopped(self : IncrementalOutput) -> Bool {
  self.lifecycle == OUTPUT_STOPPED
}

///|
pub fn IncrementalOutput::is_finished(self : IncrementalOutput) -> Bool {
  self.lifecycle == OUTPUT_FINISHED
}

///|
pub fn IncrementalOutput::has_string_stops(self : IncrementalOutput) -> Bool {
  self.patterns.length() != 0
}

///|
fn destination_available(
  destination : FixedArray[Byte],
  offset : Int,
  required : Int,
) -> Unit raise IncrementalOutputError {
  if offset < 0 ||
    offset > destination.length() ||
    required > destination.length() - offset {
    raise invalid(Destination, TooSmall)
  }
}

///|
fn copy_range(
  source : FixedArray[Byte],
  source_offset : Int,
  destination : FixedArray[Byte],
  destination_offset : Int,
  count : Int,
) -> Unit {
  for index in 0..<count {
    destination[destination_offset + index] = source[source_offset + index]
  }
}

///|
fn IncrementalOutput::copy_piece(
  self : IncrementalOutput,
  token_id : Int,
) -> Int raise IncrementalOutputError {
  self.tokenizer.copy_decoded_piece(
    token_id,
    self.scratch,
    destination_offset=self.pending_count,
    special_policy=self.special_policy,
  ) catch {
    UnknownTokenId(..) => raise invalid(Token, Invalid)
    _ => raise invalid(DecodeEnvelope, Invalid)
  }
}

///|
/// Incrementally decodes one generated token into caller-owned storage.
/// Rejection is transactional for the logical stream state and destination.
/// A matched stop string is withheld; bytes following it in the same token
/// piece are ignored because the stream is terminal.
pub fn IncrementalOutput::push_token_into(
  self : IncrementalOutput,
  token_id : Int,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
) -> IncrementalOutputStep raise IncrementalOutputError {
  if self.lifecycle != OUTPUT_ACTIVE {
    raise invalid(Lifecycle, Terminal)
  }
  copy_range(self.pending, 0, self.scratch, 0, self.pending_count)
  let piece_count = self.copy_piece(token_id)
  let total = self.pending_count + piece_count
  for index in 0..<self.states.length() {
    self.next_states[index] = self.states[index]
  }
  let mut remaining = self.utf8_remaining
  let mut codepoint = self.utf8_codepoint
  let mut minimum = self.utf8_minimum
  let mut seen = self.utf8_seen
  let mut match_end = -1
  let mut match_length = 0
  let mut cursor = self.pending_count
  while cursor < total && match_end < 0 {
    let advanced = advance_utf8(
      remaining,
      codepoint,
      minimum,
      seen,
      self.scratch[cursor],
    )
    remaining = advanced.0
    codepoint = advanced.1
    minimum = advanced.2
    seen = advanced.3
    for pattern_index, pattern in self.patterns {
      let next = pattern.advance(
        self.next_states[pattern_index],
        self.scratch[cursor],
      )
      if next == pattern.bytes.length() {
        if pattern.bytes.length() > match_length {
          match_length = pattern.bytes.length()
        }
      } else {
        self.next_states[pattern_index] = next
      }
    }
    if match_length > 0 {
      match_end = cursor
    }
    cursor += 1
  }
  if match_end >= 0 {
    if remaining != 0 {
      raise invalid(Utf8, Invalid)
    }
    let output_count = match_end + 1 - match_length
    destination_available(destination, destination_offset, output_count)
    copy_range(self.scratch, 0, destination, destination_offset, output_count)
    self.pending_count = 0
    self.utf8_remaining = 0
    self.utf8_codepoint = 0
    self.utf8_minimum = 0
    self.utf8_seen = 0
    self.lifecycle = OUTPUT_STOPPED
    return IncrementalOutputStep(output_count, true)
  }
  let mut retained = if remaining == 0 { 0 } else { seen }
  for state in self.next_states {
    if state > retained {
      retained = state
    }
  }
  let output_count = total - retained
  destination_available(destination, destination_offset, output_count)
  copy_range(self.scratch, 0, destination, destination_offset, output_count)
  copy_range(self.scratch, output_count, self.pending, 0, retained)
  for index in 0..<self.states.length() {
    self.states[index] = self.next_states[index]
  }
  self.pending_count = retained
  self.utf8_remaining = remaining
  self.utf8_codepoint = codepoint
  self.utf8_minimum = minimum
  self.utf8_seen = seen
  IncrementalOutputStep(output_count, false)
}

///|
/// Flushes an unmatched stop prefix after the final token. Truncated UTF-8 is
/// rejected and remains retryable only by supplying more token bytes.
pub fn IncrementalOutput::finish_into(
  self : IncrementalOutput,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
) -> Int raise IncrementalOutputError {
  if self.lifecycle != OUTPUT_ACTIVE {
    raise invalid(Lifecycle, Terminal)
  }
  if self.utf8_remaining != 0 {
    raise invalid(Utf8, Invalid)
  }
  destination_available(destination, destination_offset, self.pending_count)
  let count = self.pending_count
  copy_range(self.pending, 0, destination, destination_offset, count)
  self.pending_count = 0
  self.lifecycle = OUTPUT_FINISHED
  count
}

///|
/// Allocation-free decode used after a session has consumed an authenticated
/// scheduler token. This narrow method accepts only the no-string-stop shape;
/// rejection is scalar and leaves logical stream/destination state unchanged.
pub fn IncrementalOutput::push_token_without_stops_status(
  self : IncrementalOutput,
  token_id : Int,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
) -> IncrementalOutputStatus {
  if self.lifecycle != OUTPUT_ACTIVE || self.patterns.length() != 0 {
    return IncrementalOutputStatus(1, 0)
  }
  copy_range(self.pending, 0, self.scratch, 0, self.pending_count)
  let piece = self.tokenizer.copy_decoded_piece_status(
    token_id,
    self.scratch,
    destination_offset=self.pending_count,
    special_policy=self.special_policy,
  )
  if !piece.is_ready() {
    return IncrementalOutputStatus(2, 0)
  }
  let total = self.pending_count + piece.copied_bytes()
  let mut remaining = self.utf8_remaining
  let mut codepoint = self.utf8_codepoint
  let mut minimum = self.utf8_minimum
  let mut seen = self.utf8_seen
  for cursor in self.pending_count..<total {
    let advanced = advance_utf8_status(
      remaining,
      codepoint,
      minimum,
      seen,
      self.scratch[cursor],
    )
    if !advanced.0 {
      return IncrementalOutputStatus(3, 0)
    }
    remaining = advanced.1
    codepoint = advanced.2
    minimum = advanced.3
    seen = advanced.4
  }
  let retained = if remaining == 0 { 0 } else { seen }
  let output_count = total - retained
  if destination_offset < 0 ||
    destination_offset > destination.length() ||
    output_count > destination.length() - destination_offset {
    return IncrementalOutputStatus(4, 0)
  }
  copy_range(self.scratch, 0, destination, destination_offset, output_count)
  copy_range(self.scratch, output_count, self.pending, 0, retained)
  self.pending_count = retained
  self.utf8_remaining = remaining
  self.utf8_codepoint = codepoint
  self.utf8_minimum = minimum
  self.utf8_seen = seen
  IncrementalOutputStatus(0, output_count)
}

///|
/// Allocation-free terminal tail flush. Failure is scalar and transactional.
pub fn IncrementalOutput::finish_into_status(
  self : IncrementalOutput,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
) -> IncrementalOutputStatus {
  if self.lifecycle != OUTPUT_ACTIVE || self.utf8_remaining != 0 {
    return IncrementalOutputStatus(1, 0)
  }
  if destination_offset < 0 ||
    destination_offset > destination.length() ||
    self.pending_count > destination.length() - destination_offset {
    return IncrementalOutputStatus(2, 0)
  }
  let count = self.pending_count
  copy_range(self.pending, 0, destination, destination_offset, count)
  self.pending_count = 0
  self.lifecycle = OUTPUT_FINISHED
  IncrementalOutputStatus(0, count)
}

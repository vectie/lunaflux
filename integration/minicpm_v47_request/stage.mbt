///|
/// Build metadata into retained buffers. No request progress commits until
/// the submitted frame completes; intermediate prefill chunks skip logits.
pub fn Host::stage_next(self : Self) -> Bool raise {
  if self.state is Finished {
    return false
  }
  guard self.state is Ready else { raise RequestError::InvalidLifecycle }
  let prompt = self.prompt.unwrap()
  let prefill = self.cursor < prompt.tokens().length()
  let rows = if prefill {
    self.profile.rows.min(prompt.tokens().length() - self.cursor)
  } else {
    1
  }
  let end_position = self.cached + rows
  guard end_position <= self.ceiling else { raise RequestError::InvalidRequest }
  if prefill {
    for row in 0..<rows {
      write_i32(self.packet(TokenIds), row, prompt.tokens()[self.cursor + row])
    }
    prompt
    .positions()
    .write_prompt_into(start=self.cursor, rows~, self.canvas, self.causal)
  } else {
    write_i32(self.packet(TokenIds), 0, self.previous)
    prompt
    .positions()
    .write_decode_into(
      causal_start=self.cached,
      rows=1,
      self.canvas,
      self.causal,
    )
  }
  for row in 0..<rows {
    write_i32(self.packet(CausalPositions), row, self.causal[row])
    for axis in 0..<3 {
      write_i32(
        self.packet(CanvasPositions),
        row * 3 + axis,
        self.canvas[row * 3 + axis],
      )
    }
  }
  let pages = ((
    end_position.to_int64() + self.profile.page_tokens.to_int64() - 1L
  ) /
  self.profile.page_tokens.to_int64()).to_int()
  let counts = self.packet(Counts)
  write_i32(counts, 0, if prefill { 1 } else { 0 })
  write_i32(counts, 1, if prefill { 0 } else { 1 })
  write_i32(counts, 2, 1)
  write_i32(counts, 3, rows)
  write_i32(counts, 4, pages)
  write_i32(self.packet(RowOffsets), 1, rows)
  write_i32(self.packet(SequenceLength), 0, end_position)
  write_i32(self.packet(PageOffsets), 1, pages)
  write_i32(self.packet(Reset), 0, if self.cached == 0 { 1 } else { 0 })
  let needs_token = !prefill || self.cursor + rows == prompt.tokens().length()
  write_i32(self.packet(OutputCount), 0, if needs_token { 1 } else { 0 })
  write_i32(self.packet(SelectedRows), 0, rows - 1)
  self.phase = if prefill { Prefill } else { Decode }
  self.rows = rows
  self.end_position = end_position
  self.needs_token = needs_token
  self.state = Staged
  true
}

///|
pub fn Host::mark_submitted(self : Self) -> Unit raise RequestError {
  guard self.state is Staged else { raise InvalidLifecycle }
  self.state = Submitted
}

///|
/// Call after the device completion barrier. Token is ignored when this is an
/// intermediate prefill frame. Invalid sampled values fail the request.
pub fn Host::complete_step(
  self : Self,
  token : Int32,
) -> Unit raise RequestError {
  guard self.state is Submitted else { raise InvalidLifecycle }
  if self.needs_token && (token < 0 || token >= self.profile.vocabulary) {
    self.state = Failed
    raise InvalidToken
  }
  self.cached = self.end_position
  if self.phase is Prefill {
    self.cursor += self.rows
  }
  self.state = Ready
  if !self.needs_token {
    return
  }
  self.sampled += 1
  if self.stops.contains(token) {
    self.reason = Some(StopSequence)
    self.state = Finished
    return
  }
  self.output[self.emitted] = token
  self.emitted += 1
  self.previous = token
  if self.sampled >= self.maximum_new || self.cached >= self.ceiling {
    self.reason = Some(TokenLimit)
    self.state = Finished
  }
}

///|
/// Caller must retire/abort device work before reusing packets or starting the
/// next request. The native driver provides that barrier before this method.
pub fn Host::stop_quiescent(
  self : Self,
  reason : @inference.FinishReason,
) -> Unit {
  self.reason = Some(reason)
  self.state = Finished
}

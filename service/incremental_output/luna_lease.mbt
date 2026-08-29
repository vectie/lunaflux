///|
fn LunaIncrementalOutputLease::require_workspace(
  self : LunaIncrementalOutputLease,
) -> LunaIncrementalOutputWorkspace raise IncrementalOutputError {
  if self.epoch == 0UL ||
    self.workspace.epoch != self.epoch ||
    self.workspace.phase != LUNA_INCREMENTAL_OUTPUT_LEASED {
    raise invalid(LunaIncrementalOutputEpoch, Stale)
  }
  self.workspace
}

///|
fn pattern_advance(
  workspace : LunaIncrementalOutputWorkspace,
  pattern : Int,
  state : Int,
  byte : Byte,
) -> Int {
  let capacity = workspace.stop_string_byte_capacity
  let base = pattern * capacity
  let mut next = state
  while next > 0 && workspace.pattern_bytes[base + next] != byte {
    next = workspace.failure[base + next - 1]
  }
  if workspace.pattern_bytes[base + next] == byte {
    next += 1
  }
  next
}

///|
fn LunaIncrementalOutputWorkspace::push_token_status_current(
  self : LunaIncrementalOutputWorkspace,
  token_id : Int,
  destination : FixedArray[Byte],
  destination_offset : Int,
) -> IncrementalOutputStatus {
  if self.lifecycle != OUTPUT_ACTIVE {
    return IncrementalOutputStatus(1, 0, false)
  }
  copy_range(self.pending, 0, self.scratch, 0, self.pending_count)
  let piece = self.tokenizer.copy_decoded_piece_status(
    token_id,
    self.scratch,
    destination_offset=self.pending_count,
    special_policy=self.special_policy,
  )
  if !piece.is_ready() {
    return IncrementalOutputStatus(2, 0, false)
  }
  let mut piece_bytes = piece.copied_bytes()
  let mut next_decoder_at_start = self.decoder_at_start
  if next_decoder_at_start && piece_bytes > 0 {
    next_decoder_at_start = false
    if self.tokenizer.decoder_strips_initial_space() &&
      self.scratch[self.pending_count] == b' ' {
      for index in 1..<piece_bytes {
        self.scratch[self.pending_count + index - 1] = self.scratch[self.pending_count +
          index]
      }
      piece_bytes -= 1
    }
  }
  let total = self.pending_count + piece_bytes
  for index in 0..<self.stop_count {
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
    let advanced = advance_utf8_status(
      remaining,
      codepoint,
      minimum,
      seen,
      self.scratch[cursor],
    )
    if !advanced.0 {
      return IncrementalOutputStatus(3, 0, false)
    }
    remaining = advanced.1
    codepoint = advanced.2
    minimum = advanced.3
    seen = advanced.4
    for pattern_index in 0..<self.stop_count {
      let next = pattern_advance(
        self,
        pattern_index,
        self.next_states[pattern_index],
        self.scratch[cursor],
      )
      if next == self.pattern_lengths[pattern_index] {
        if self.pattern_lengths[pattern_index] > match_length {
          match_length = self.pattern_lengths[pattern_index]
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
      return IncrementalOutputStatus(3, 0, false)
    }
    let output_count = match_end + 1 - match_length
    if destination_offset < 0 ||
      destination_offset > destination.length() ||
      output_count > destination.length() - destination_offset {
      return IncrementalOutputStatus(4, 0, false)
    }
    copy_range(self.scratch, 0, destination, destination_offset, output_count)
    self.pending_count = 0
    self.decoder_at_start = next_decoder_at_start
    self.utf8_remaining = 0
    self.utf8_codepoint = 0
    self.utf8_minimum = 0
    self.utf8_seen = 0
    self.lifecycle = OUTPUT_STOPPED
    return IncrementalOutputStatus(0, output_count, true)
  }
  let mut retained = if remaining == 0 { 0 } else { seen }
  for index in 0..<self.stop_count {
    if self.next_states[index] > retained {
      retained = self.next_states[index]
    }
  }
  let output_count = total - retained
  if destination_offset < 0 ||
    destination_offset > destination.length() ||
    output_count > destination.length() - destination_offset {
    return IncrementalOutputStatus(4, 0, false)
  }
  copy_range(self.scratch, 0, destination, destination_offset, output_count)
  copy_range(self.scratch, output_count, self.pending, 0, retained)
  for index in 0..<self.stop_count {
    self.states[index] = self.next_states[index]
  }
  self.pending_count = retained
  self.decoder_at_start = next_decoder_at_start
  self.utf8_remaining = remaining
  self.utf8_codepoint = codepoint
  self.utf8_minimum = minimum
  self.utf8_seen = seen
  IncrementalOutputStatus(0, output_count, false)
}

///|
pub fn LunaIncrementalOutputLease::push_token_status(
  self : LunaIncrementalOutputLease,
  token_id : Int,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
) -> IncrementalOutputStatus {
  let workspace = try self.require_workspace() catch {
    _ => return IncrementalOutputStatus::rejected()
  } noraise {
    value => value
  }
  workspace.push_token_status_current(token_id, destination, destination_offset)
}

///|
pub fn LunaIncrementalOutputLease::push_token_without_stops_status(
  self : LunaIncrementalOutputLease,
  token_id : Int,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
) -> IncrementalOutputStatus {
  let workspace = try self.require_workspace() catch {
    _ => return IncrementalOutputStatus::rejected()
  } noraise {
    value => value
  }
  if workspace.stop_count != 0 {
    return IncrementalOutputStatus::rejected()
  }
  workspace.push_token_status_current(token_id, destination, destination_offset)
}

///|
pub fn LunaIncrementalOutputLease::finish_into_status(
  self : LunaIncrementalOutputLease,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
) -> IncrementalOutputStatus {
  let workspace = try self.require_workspace() catch {
    _ => return IncrementalOutputStatus::rejected()
  } noraise {
    value => value
  }
  if workspace.lifecycle != OUTPUT_ACTIVE || workspace.utf8_remaining != 0 {
    return IncrementalOutputStatus(1, 0, false)
  }
  if destination_offset < 0 ||
    destination_offset > destination.length() ||
    workspace.pending_count > destination.length() - destination_offset {
    return IncrementalOutputStatus(2, 0, false)
  }
  let count = workspace.pending_count
  copy_range(workspace.pending, 0, destination, destination_offset, count)
  workspace.pending_count = 0
  workspace.decoder_at_start = true
  workspace.lifecycle = OUTPUT_FINISHED
  IncrementalOutputStatus(0, count, false)
}

///|
pub fn LunaIncrementalOutputLease::is_stopped(
  self : LunaIncrementalOutputLease,
) -> Bool raise IncrementalOutputError {
  self.require_workspace().lifecycle == OUTPUT_STOPPED
}

///|
pub fn LunaIncrementalOutputLease::is_finished(
  self : LunaIncrementalOutputLease,
) -> Bool raise IncrementalOutputError {
  self.require_workspace().lifecycle == OUTPUT_FINISHED
}

///|
pub fn LunaIncrementalOutputLease::release(
  self : LunaIncrementalOutputLease,
) -> Unit raise IncrementalOutputError {
  let workspace = self.require_workspace()
  workspace.phase = LUNA_INCREMENTAL_OUTPUT_IDLE
  workspace.stop_count = 0
  workspace.pending_count = 0
  workspace.decoder_at_start = true
  workspace.last_work_units = 0
  workspace.lifecycle = OUTPUT_ACTIVE
}

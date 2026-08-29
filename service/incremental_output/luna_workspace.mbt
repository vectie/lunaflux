///|
pub fn LunaIncrementalOutputStepBudget::new(
  work_units : Int,
) -> LunaIncrementalOutputStepBudget raise IncrementalOutputError {
  if work_units <= 0 || work_units > LUNA_INCREMENTAL_OUTPUT_MAX_STEP_WORK_UNITS {
    raise invalid(LunaIncrementalOutputWorkspace, Unsupported)
  }
  { work_units, }
}

///|
pub fn LunaIncrementalOutputStepBudget::work_units(
  self : LunaIncrementalOutputStepBudget,
) -> Int {
  self.work_units
}

///|
pub fn LunaIncrementalOutputWorkspace::begin(
  self : LunaIncrementalOutputWorkspace,
  stops : @inference.StopConditions,
  special_policy~ : @tokenizer.DecodeSpecialPolicy,
) -> LunaIncrementalOutputWork raise IncrementalOutputError {
  self.begin_setup(stops.strings().length(), Some(stops), special_policy)
}

///|
fn LunaIncrementalOutputWorkspace::begin_setup(
  self : LunaIncrementalOutputWorkspace,
  stop_count : Int,
  setup_stops : @inference.StopConditions?,
  special_policy : @tokenizer.DecodeSpecialPolicy,
) -> LunaIncrementalOutputWork raise IncrementalOutputError {
  if self.phase != LUNA_INCREMENTAL_OUTPUT_IDLE {
    raise invalid(LunaIncrementalOutputWorkspace, Busy)
  }
  if self.epoch == 0xffffffffffffffffUL {
    raise invalid(LunaIncrementalOutputEpoch, Exhausted)
  }
  if stop_count < 0 || stop_count > self.pattern_lengths.length() {
    raise invalid(LunaIncrementalOutputWorkspace, Unsupported)
  }
  self.epoch += 1UL
  self.setup_stops = setup_stops
  self.special_policy = special_policy
  self.stop_count = stop_count
  self.setup_pattern = 0
  self.setup_unit_offset = 0
  self.setup_emit_index = 0
  self.setup_emit_length = 0
  self.setup_codepoint = 0
  self.setup_failure_index = 0
  self.setup_failure_matched = 0
  self.longest_pattern = 0
  self.last_work_units = 0
  self.setup_error_rule = Lifecycle
  self.setup_error_issue = Invalid
  self.pending_count = 0
  self.decoder_at_start = true
  self.utf8_remaining = 0
  self.utf8_codepoint = 0
  self.utf8_minimum = 0
  self.utf8_seen = 0
  self.lifecycle = OUTPUT_ACTIVE
  self.phase = LUNA_INCREMENTAL_OUTPUT_SETUP_PATTERN
  { workspace: self, epoch: self.epoch }
}

///|
fn LunaIncrementalOutputWork::require_setup(
  self : LunaIncrementalOutputWork,
) -> LunaIncrementalOutputWorkspace raise IncrementalOutputError {
  if self.epoch == 0UL ||
    self.workspace.epoch != self.epoch ||
    (
      self.workspace.phase != LUNA_INCREMENTAL_OUTPUT_SETUP_PATTERN &&
      self.workspace.phase != LUNA_INCREMENTAL_OUTPUT_SETUP_TEXT &&
      self.workspace.phase != LUNA_INCREMENTAL_OUTPUT_SETUP_FAILURE &&
      self.workspace.phase != LUNA_INCREMENTAL_OUTPUT_READY &&
      self.workspace.phase != LUNA_INCREMENTAL_OUTPUT_FAILED
    ) {
    raise invalid(LunaIncrementalOutputEpoch, Stale)
  }
  self.workspace
}

///|
fn LunaIncrementalOutputWorkspace::clear_setup_source(
  self : LunaIncrementalOutputWorkspace,
) -> Unit {
  self.setup_stops = None
  self.setup_semantics.clear()
}

///|
fn LunaIncrementalOutputWorkspace::fail_setup(
  self : LunaIncrementalOutputWorkspace,
  rule : IncrementalOutputRule,
  issue : IncrementalOutputIssue,
) -> Unit {
  self.clear_setup_source()
  self.setup_error_rule = rule
  self.setup_error_issue = issue
  self.phase = LUNA_INCREMENTAL_OUTPUT_FAILED
}

///|
fn LunaIncrementalOutputWorkspace::raise_setup_failure(
  self : LunaIncrementalOutputWorkspace,
) -> Unit raise IncrementalOutputError {
  raise invalid(self.setup_error_rule, self.setup_error_issue)
}

///|
fn utf8_width(codepoint : Int) -> Int {
  if codepoint <= 0x7f {
    1
  } else if codepoint <= 0x7ff {
    2
  } else if codepoint <= 0xffff {
    3
  } else {
    4
  }
}

///|
fn utf8_byte(codepoint : Int, width : Int, index : Int) -> Byte {
  if width == 1 {
    codepoint.to_byte()
  } else if index == 0 {
    ((0xf0 << (4 - width)) | (codepoint >> (6 * (width - 1)))).to_byte()
  } else {
    (0x80 | ((codepoint >> (6 * (width - index - 1))) & 0x3f)).to_byte()
  }
}

///|
fn LunaIncrementalOutputWorkspace::progress_setup_pattern(
  self : LunaIncrementalOutputWorkspace,
) -> Unit raise IncrementalOutputError {
  if self.setup_pattern >= self.stop_count {
    let token_bytes = self.tokenizer.limits().max_token_bytes
    let delta_bytes = self.inference.max_decoded_delta_bytes()
    if self.longest_pattern > delta_bytes - token_bytes - 3 {
      self.fail_setup(DecodeEnvelope, Unsupported)
      self.raise_setup_failure()
    }
    self.authenticate_setup_semantics()
    self.phase = LUNA_INCREMENTAL_OUTPUT_READY
    return
  }
  self.pattern_lengths[self.setup_pattern] = 0
  self.states[self.setup_pattern] = 0
  self.next_states[self.setup_pattern] = 0
  self.setup_unit_offset = 0
  self.setup_emit_index = 0
  self.setup_emit_length = 0
  self.setup_codepoint = 0
  self.setup_failure_index = 0
  self.setup_failure_matched = 0
  self.phase = LUNA_INCREMENTAL_OUTPUT_SETUP_TEXT
}

///|
fn LunaIncrementalOutputWorkspace::progress_setup_text(
  self : LunaIncrementalOutputWorkspace,
) -> Unit raise IncrementalOutputError {
  if !self.setup_semantics.is_empty() {
    self.progress_setup_semantic_text()
    return
  }
  let value = self.setup_stops.unwrap().strings()[self.setup_pattern]
  let maximum = self.stop_string_byte_capacity
  if self.setup_emit_index < self.setup_emit_length {
    let length = self.pattern_lengths[self.setup_pattern]
    if length >= maximum {
      self.fail_setup(DecodeEnvelope, Unsupported)
      self.raise_setup_failure()
    }
    let base = self.setup_pattern * maximum
    self.pattern_bytes[base + length] = utf8_byte(
      self.setup_codepoint,
      self.setup_emit_length,
      self.setup_emit_index,
    )
    self.pattern_lengths[self.setup_pattern] = length + 1
    self.setup_emit_index += 1
    return
  }
  if self.setup_unit_offset >= value.length() {
    let length = self.pattern_lengths[self.setup_pattern]
    if length <= 0 {
      abort("validated stop string length invariant")
    }
    self.failure[self.setup_pattern * maximum] = 0
    self.setup_failure_index = 1
    self.setup_failure_matched = 0
    self.phase = LUNA_INCREMENTAL_OUTPUT_SETUP_FAILURE
    return
  }
  let first = value.unsafe_get(self.setup_unit_offset).to_int()
  if first >= 0xd800 && first <= 0xdbff {
    if self.setup_unit_offset + 1 >= value.length() {
      abort("validated stop string surrogate invariant")
    }
    let second = value.unsafe_get(self.setup_unit_offset + 1).to_int()
    if second < 0xdc00 || second > 0xdfff {
      abort("validated stop string surrogate invariant")
    }
    self.setup_codepoint = 0x10000 +
      ((first - 0xd800) << 10) +
      (second - 0xdc00)
    self.setup_unit_offset += 2
  } else {
    if first >= 0xdc00 && first <= 0xdfff {
      abort("validated stop string surrogate invariant")
    }
    self.setup_codepoint = first
    self.setup_unit_offset += 1
  }
  self.setup_emit_length = utf8_width(self.setup_codepoint)
  self.setup_emit_index = 0
}

///|
fn LunaIncrementalOutputWorkspace::progress_setup_failure(
  self : LunaIncrementalOutputWorkspace,
) -> Unit {
  let capacity = self.stop_string_byte_capacity
  let base = self.setup_pattern * capacity
  let length = self.pattern_lengths[self.setup_pattern]
  if self.setup_failure_index >= length {
    if length > self.longest_pattern {
      self.longest_pattern = length
    }
    self.setup_pattern += 1
    self.phase = LUNA_INCREMENTAL_OUTPUT_SETUP_PATTERN
    return
  }
  let current = self.pattern_bytes[base + self.setup_failure_index]
  if self.setup_failure_matched > 0 &&
    current != self.pattern_bytes[base + self.setup_failure_matched] {
    self.setup_failure_matched = self.failure[base +
      self.setup_failure_matched -
      1]
    return
  }
  if current == self.pattern_bytes[base + self.setup_failure_matched] {
    self.setup_failure_matched += 1
  }
  self.failure[base + self.setup_failure_index] = self.setup_failure_matched
  self.setup_failure_index += 1
}

///|
fn LunaIncrementalOutputWorkspace::progress_setup_one(
  self : LunaIncrementalOutputWorkspace,
) -> Unit raise IncrementalOutputError {
  match self.phase {
    LUNA_INCREMENTAL_OUTPUT_SETUP_PATTERN => self.progress_setup_pattern()
    LUNA_INCREMENTAL_OUTPUT_SETUP_TEXT => self.progress_setup_text()
    LUNA_INCREMENTAL_OUTPUT_SETUP_FAILURE => self.progress_setup_failure()
    _ => abort("Luna incremental output setup phase invariant")
  }
}

///|
pub fn LunaIncrementalOutputWork::progress(
  self : LunaIncrementalOutputWork,
) -> LunaIncrementalOutputProgress raise IncrementalOutputError {
  let workspace = self.require_setup()
  if workspace.phase == LUNA_INCREMENTAL_OUTPUT_FAILED {
    workspace.last_work_units = 0
    workspace.raise_setup_failure()
  }
  if workspace.phase == LUNA_INCREMENTAL_OUTPUT_READY {
    workspace.last_work_units = 0
    return LunaIncrementalOutputReady
  }
  let mut used = 0
  while used < workspace.budget.work_units() &&
        workspace.phase != LUNA_INCREMENTAL_OUTPUT_READY {
    workspace.progress_setup_one() catch {
      error => {
        workspace.last_work_units = used + 1
        raise error
      }
    }
    used += 1
  }
  workspace.last_work_units = used
  if workspace.phase == LUNA_INCREMENTAL_OUTPUT_READY {
    LunaIncrementalOutputReady
  } else {
    LunaIncrementalOutputPending
  }
}

///|
pub fn LunaIncrementalOutputWork::last_work_units(
  self : LunaIncrementalOutputWork,
) -> Int raise IncrementalOutputError {
  self.require_setup().last_work_units
}

///|
pub fn LunaIncrementalOutputWork::take_lease(
  self : LunaIncrementalOutputWork,
) -> LunaIncrementalOutputLease raise IncrementalOutputError {
  let workspace = self.require_setup()
  if workspace.phase != LUNA_INCREMENTAL_OUTPUT_READY {
    raise invalid(LunaIncrementalOutputWorkspace, NotReady)
  }
  workspace.authenticate_setup_semantics()
  workspace.phase = LUNA_INCREMENTAL_OUTPUT_LEASED
  workspace.clear_setup_source()
  { workspace, epoch: self.epoch }
}

///|
pub fn LunaIncrementalOutputWork::abort(
  self : LunaIncrementalOutputWork,
) -> Unit raise IncrementalOutputError {
  let workspace = self.require_setup()
  workspace.phase = LUNA_INCREMENTAL_OUTPUT_IDLE
  workspace.clear_setup_source()
  workspace.setup_error_rule = Lifecycle
  workspace.setup_error_issue = Invalid
  workspace.last_work_units = 0
}

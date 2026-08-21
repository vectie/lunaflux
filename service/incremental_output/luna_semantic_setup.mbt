///|
pub fn LunaIncrementalOutputWorkspace::begin_luna_request_semantics(
  self : LunaIncrementalOutputWorkspace,
  semantics : @inference.LunaRequestSemanticView,
  special_policy~ : @tokenizer.DecodeSpecialPolicy,
) -> LunaIncrementalOutputWork raise IncrementalOutputError {
  if self.phase != LUNA_INCREMENTAL_OUTPUT_IDLE {
    raise invalid(LunaIncrementalOutputWorkspace, Busy)
  }
  if self.epoch == 0xffffffffffffffffUL {
    raise invalid(LunaIncrementalOutputEpoch, Exhausted)
  }
  let semantic_limits = semantics.inference_limits() catch {
    _ => raise invalid(LunaIncrementalOutputEpoch, Stale)
  }
  if semantic_limits != self.inference {
    raise invalid(DecodeEnvelope, Unsupported)
  }
  let stop_count = semantics.stop_string_count() catch {
    _ => raise invalid(LunaIncrementalOutputEpoch, Stale)
  }
  if stop_count < 0 || stop_count > self.pattern_lengths.length() {
    raise invalid(LunaIncrementalOutputWorkspace, Unsupported)
  }
  self.setup_semantics.push(semantics)
  self.begin_setup(stop_count, None, special_policy) catch {
    error => {
      self.setup_semantics.clear()
      raise error
    }
  }
}

///|
fn LunaIncrementalOutputWorkspace::authenticate_setup_semantics(
  self : LunaIncrementalOutputWorkspace,
) -> Unit raise IncrementalOutputError {
  if !self.setup_semantics.is_empty() {
    let semantics = self.setup_semantics[0]
    let semantic_limits = semantics.inference_limits() catch {
      _ => {
        self.fail_setup(LunaIncrementalOutputEpoch, Stale)
        raise invalid(self.setup_error_rule, self.setup_error_issue)
      }
    }
    if semantic_limits != self.inference {
      self.fail_setup(DecodeEnvelope, Unsupported)
      raise invalid(self.setup_error_rule, self.setup_error_issue)
    }
  }
}

///|
fn LunaIncrementalOutputWorkspace::semantic_stop_string_length(
  self : LunaIncrementalOutputWorkspace,
  semantics : @inference.LunaRequestSemanticView,
  index : Int,
) -> Int raise IncrementalOutputError {
  semantics.stop_string_length(index) catch {
    _ => {
      self.fail_setup(LunaIncrementalOutputEpoch, Stale)
      raise invalid(self.setup_error_rule, self.setup_error_issue)
    }
  }
}

///|
fn LunaIncrementalOutputWorkspace::semantic_stop_string_byte_at(
  self : LunaIncrementalOutputWorkspace,
  semantics : @inference.LunaRequestSemanticView,
  string_index : Int,
  byte_index : Int,
) -> Byte raise IncrementalOutputError {
  semantics.stop_string_byte_at(string_index~, byte_index~) catch {
    _ => {
      self.fail_setup(LunaIncrementalOutputEpoch, Stale)
      raise invalid(self.setup_error_rule, self.setup_error_issue)
    }
  }
}

///|
fn LunaIncrementalOutputWorkspace::progress_setup_semantic_text(
  self : LunaIncrementalOutputWorkspace,
) -> Unit raise IncrementalOutputError {
  let semantics = self.setup_semantics[0]
  let length = self.semantic_stop_string_length(semantics, self.setup_pattern)
  if self.setup_unit_offset >= length {
    let copied = self.pattern_lengths[self.setup_pattern]
    if copied <= 0 {
      abort("validated semantic stop string length invariant")
    }
    let base = self.setup_pattern * self.stop_string_byte_capacity
    self.failure[base] = 0
    self.setup_failure_index = 1
    self.setup_failure_matched = 0
    self.phase = LUNA_INCREMENTAL_OUTPUT_SETUP_FAILURE
    return
  }
  let copied = self.pattern_lengths[self.setup_pattern]
  if copied >= self.stop_string_byte_capacity {
    self.fail_setup(DecodeEnvelope, Unsupported)
    self.raise_setup_failure()
  }
  let base = self.setup_pattern * self.stop_string_byte_capacity
  self.pattern_bytes[base + copied] = self.semantic_stop_string_byte_at(
    semantics,
    self.setup_pattern,
    self.setup_unit_offset,
  )
  self.pattern_lengths[self.setup_pattern] = copied + 1
  self.setup_unit_offset += 1
}

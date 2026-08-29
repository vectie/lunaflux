///|
fn luna_text_u32_byte(value : UInt, index : Int) -> Byte {
  (value >> (index * 8)).to_byte()
}

///|
fn luna_text_u64_byte(value : UInt64, index : Int) -> Byte {
  (value >> (index * 8)).to_byte()
}

///|
fn LunaFramedTextRequestWorkspace::flags(self : Self) -> UInt {
  1U |
  (if self.sampling_mode == 1 { 2U } else { 0U }) |
  (if self.top_k != 0 { 4U } else { 0U }) |
  (if self.top_p_bits != 0UL { 8U } else { 0U }) |
  (if self.stream { 16U } else { 0U })
}

///|
fn LunaFramedTextRequestWorkspace::header_byte(
  self : Self,
  index : Int,
) -> Byte {
  let length = REQUEST_HEADER_BYTES +
    self.input_length +
    self.stop_token_count * 4 +
    self.stop_string_count * 4 +
    self.stop_string_bytes +
    self.cache_scope_length
  if index < 4 {
    return luna_text_u32_byte(request_magic, index)
  }
  if index < 8 {
    return luna_text_u32_byte(request_version, index - 4)
  }
  if index < 12 {
    return luna_text_u32_byte(1U, index - 8)
  }
  if index < 16 {
    return luna_text_u32_byte(length.reinterpret_as_uint(), index - 12)
  }
  if index < 20 {
    return 0
  }
  if index < 24 {
    return luna_text_u32_byte(self.flags(), index - 20)
  }
  if index < 28 {
    return luna_text_u32_byte(
      self.protocol_version.reinterpret_as_uint(),
      index - 24,
    )
  }
  if index < 32 {
    return luna_text_u32_byte(
      self.input_length.reinterpret_as_uint(),
      index - 28,
    )
  }
  if index < 44 {
    if index < 36 {
      return luna_text_u32_byte(
        self.stop_token_count.reinterpret_as_uint(),
        index - 32,
      )
    }
    if index < 40 {
      return luna_text_u32_byte(
        self.stop_string_count.reinterpret_as_uint(),
        index - 36,
      )
    }
    return luna_text_u32_byte(
      self.stop_string_bytes.reinterpret_as_uint(),
      index - 40,
    )
  }
  if index < 48 {
    return luna_text_u32_byte(
      self.cache_scope_length.reinterpret_as_uint(),
      index - 44,
    )
  }
  if index < 52 {
    return 0
  }
  if index < 56 {
    return luna_text_u32_byte(
      self.max_new_tokens.reinterpret_as_uint(),
      index - 52,
    )
  }
  if index < 60 {
    return luna_text_u32_byte(
      self.context_ceiling.reinterpret_as_uint(),
      index - 56,
    )
  }
  if index < 64 {
    return luna_text_u32_byte(
      self.sampling_mode.reinterpret_as_uint(),
      index - 60,
    )
  }
  if index < 68 {
    return luna_text_u32_byte(self.top_k.reinterpret_as_uint(), index - 64)
  }
  if index < 72 {
    return 0
  }
  if index < 80 {
    return luna_text_u64_byte(self.temperature_bits, index - 72)
  }
  if index < 88 {
    return luna_text_u64_byte(self.top_p_bits, index - 80)
  }
  if index < 96 {
    return luna_text_u64_byte(self.sampling_seed, index - 88)
  }
  if index < 104 {
    return luna_text_u64_byte(self.deadline_millis, index - 96)
  }
  if index < 112 {
    return luna_text_u64_byte(self.request_id, index - 104)
  }
  if index < 176 {
    return self.content_digest[index - 112]
  }
  if index < 240 {
    return self.plan_digest[index - 176]
  }
  if index < 244 {
    return 0
  }
  if index < 248 {
    return luna_text_u32_byte(
      (self.input_length +
      self.stop_token_count * 4 +
      self.stop_string_count * 4 +
      self.stop_string_bytes +
      self.cache_scope_length).reinterpret_as_uint(),
      index - 244,
    )
  }
  if index < 252 {
    return luna_text_u32_byte(
      REQUEST_HEADER_BYTES.reinterpret_as_uint(),
      index - 248,
    )
  }
  0
}

///|
fn LunaFramedTextRequestWorkspace::progress_one(self : Self) -> Unit {
  let length = REQUEST_HEADER_BYTES +
    self.input_length +
    self.stop_token_count * 4 +
    self.stop_string_count * 4 +
    self.stop_string_bytes +
    self.cache_scope_length
  match self.phase {
    LUNA_TEXT_HEADER => {
      self.storage[self.header_cursor] = self.header_byte(self.header_cursor)
      self.header_cursor += 1
      if self.header_cursor == REQUEST_HEADER_BYTES {
        self.phase = LUNA_TEXT_SCOPE
      }
    }
    LUNA_TEXT_SCOPE => {
      self.storage[self.payload_cursor + self.scope_cursor] = self.cache_scope[self.scope_cursor]
      self.scope_cursor += 1
      if self.scope_cursor == self.cache_scope_length {
        self.phase = LUNA_TEXT_CHECKSUM
      }
    }
    LUNA_TEXT_CHECKSUM => {
      if self.checksum_cursor < 16 || self.checksum_cursor >= 20 {
        self.checksum = (
            self.checksum ^ self.storage[self.checksum_cursor].to_uint()
          ) *
          16777619
      }
      self.checksum_cursor += 1
      if self.checksum_cursor == length {
        self.phase = LUNA_TEXT_CHECKSUM_WRITE
      }
    }
    LUNA_TEXT_CHECKSUM_WRITE => {
      self.storage[16 + self.checksum_write_cursor] = luna_text_u32_byte(
        self.checksum,
        self.checksum_write_cursor,
      )
      self.checksum_write_cursor += 1
      if self.checksum_write_cursor == 4 {
        self.phase = LUNA_TEXT_FINAL
      }
    }
    LUNA_TEXT_FINAL => self.phase = LUNA_TEXT_READY
    _ => abort("Luna framed text progress invariant")
  }
}

///|
fn LunaFramedTextRequestWork::require(
  self : Self,
) -> LunaFramedTextRequestWorkspace raise FramedWireError {
  if self.epoch == 0UL ||
    self.workspace.epoch != self.epoch ||
    self.workspace.phase == LUNA_TEXT_IDLE ||
    self.workspace.phase == LUNA_TEXT_VIEW {
    raise LunaFramedTextRequestStale
  }
  self.workspace
}

///|
pub fn LunaFramedTextRequestWork::progress(
  self : Self,
) -> LunaFramedTextRequestProgress raise FramedWireError {
  let workspace = self.require()
  workspace.last_work_units = 0
  if workspace.phase == LUNA_TEXT_READY || workspace.phase == LUNA_TEXT_FAILED {
    return LunaFramedTextRequestIdle
  }
  let mut used = 0
  while used < workspace.budget.work_units() &&
        workspace.phase != LUNA_TEXT_READY {
    workspace.progress_one()
    used += 1
  }
  workspace.last_work_units = used
  workspace.total_work_units += used.to_uint64()
  LunaFramedTextRequestAdvanced
}

///|
pub fn LunaFramedTextRequestWork::state(
  self : Self,
) -> LunaFramedTextRequestWorkState raise FramedWireError {
  match self.require().phase {
    LUNA_TEXT_READY => LunaFramedTextRequestReady
    LUNA_TEXT_FAILED => LunaFramedTextRequestFailed
    _ => LunaFramedTextRequestPending
  }
}

///|
pub fn LunaFramedTextRequestWork::last_work_units(
  self : Self,
) -> Int raise FramedWireError {
  self.require().last_work_units
}

///|
pub fn LunaFramedTextRequestWork::total_work_units(
  self : Self,
) -> UInt64 raise FramedWireError {
  self.require().total_work_units
}

///|
pub fn LunaFramedTextRequestWork::failure(
  self : Self,
) -> FramedWireError raise FramedWireError {
  ignore(self.require())
  raise LunaFramedTextRequestNotReady
}

///|
pub fn LunaFramedTextRequestWork::abort(
  self : Self,
) -> Unit raise FramedWireError {
  let w = self.require()
  w.reset()
  w.phase = LUNA_TEXT_IDLE
}

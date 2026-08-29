///|
fn LunaOpenAICompatWorkspace::emit_byte(
  self : LunaOpenAICompatWorkspace,
  byte : Byte,
) -> Unit raise LunaOpenAICompatError {
  if self.length < 0 || self.length >= self.storage.length() {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatCapacity)
  }
  self.storage[self.length] = byte
  self.length += 1
}

///|
fn LunaOpenAICompatWorkspace::current_literal(
  self : LunaOpenAICompatWorkspace,
) -> Bytes {
  match self.part_kind {
    LUNA_OPENAI_PART_FINISH =>
      match self.finish_reason {
        0 => b"stop"
        1 => b"length"
        _ => b"cancelled"
      }
    LUNA_OPENAI_PART_RETRYABLE =>
      if self.retryable {
        b"true"
      } else {
        b"false"
      }
    _ => luna_openai_literal(self.mode, self.event_kind, self.part_index)
  }
}

///|
fn LunaOpenAICompatWorkspace::part_source_length(
  self : LunaOpenAICompatWorkspace,
) -> Int {
  if self.part_kind == LUNA_OPENAI_PART_MODEL {
    self.model_alias_length
  } else {
    self.payload_length
  }
}

///|
fn LunaOpenAICompatWorkspace::payload_byte_at(
  self : LunaOpenAICompatWorkspace,
  event : @luna_event.LunaEventView,
  index : Int,
) -> Byte raise LunaOpenAICompatError {
  match self.event_kind {
    LUNA_OPENAI_EVENT_TOKEN => {
      let token = event.token() catch {
        _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatSemanticEvent)
      }
      token.delta_byte_at(index) catch {
        _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatSemanticEvent)
      }
    }
    LUNA_OPENAI_EVENT_COMPLETED => {
      let completed = event.completed() catch {
        _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatSemanticEvent)
      }
      completed.final_delta_byte_at(index) catch {
        _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatSemanticEvent)
      }
    }
    LUNA_OPENAI_EVENT_FAILED => {
      let failed = event.failed() catch {
        _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatSemanticEvent)
      }
      failed.code_byte_at(index) catch {
        _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatSemanticEvent)
      }
    }
    _ => abort("Luna OpenAI compatibility payload byte invariant")
  }
}

///|
fn luna_openai_escape_kind(byte : Byte) -> (Int, Int) {
  match byte {
    b'"' => (1, 2)
    b'\\' => (2, 2)
    b'\x08' | b'\x09' | b'\x0a' | b'\x0c' | b'\x0d' => (3, 2)
    b'\x00'..=b'\x1f' => (4, 6)
    _ => (0, 1)
  }
}

///|
fn luna_openai_hex(value : Int) -> Byte {
  if value < 10 {
    (b'0'.to_int() + value).to_byte()
  } else {
    (b'a'.to_int() + value - 10).to_byte()
  }
}

///|
fn luna_openai_escaped_byte(kind : Int, byte : Byte, index : Int) -> Byte {
  match kind {
    0 => byte
    1 => if index == 0 { b'\\' } else { b'"' }
    2 => b'\\'
    3 =>
      if index == 0 {
        b'\\'
      } else {
        match byte {
          b'\x08' => b'b'
          b'\x09' => b't'
          b'\x0a' => b'n'
          b'\x0c' => b'f'
          _ => b'r'
        }
      }
    _ =>
      match index {
        0 => b'\\'
        1 => b'u'
        2 | 3 => b'0'
        4 => luna_openai_hex(byte.to_int() >> 4)
        _ => luna_openai_hex(byte.to_int() & 0x0f)
      }
  }
}

///|
fn LunaOpenAICompatWorkspace::decimal_for_part(
  self : LunaOpenAICompatWorkspace,
) -> UInt64 {
  match self.part_kind {
    LUNA_OPENAI_PART_SEQUENCE => self.response_sequence
    LUNA_OPENAI_PART_INPUT => self.input_tokens.to_uint64()
    LUNA_OPENAI_PART_CACHED => self.cached_input_tokens.to_uint64()
    LUNA_OPENAI_PART_OUTPUT => self.output_tokens.to_uint64()
    _ => self.total_tokens.to_uint64()
  }
}

///|
fn LunaOpenAICompatWorkspace::configure_part(
  self : LunaOpenAICompatWorkspace,
) -> Unit {
  self.part_kind = luna_openai_part_kind(
    self.mode,
    self.event_kind,
    self.part_index,
  )
  self.part_cursor = 0
  self.source_cursor = 0
  self.escape_cursor = 0
  self.escape_length = 0
  match self.part_kind {
    LUNA_OPENAI_PART_END => self.phase = LUNA_OPENAI_COMPAT_FINALIZING
    LUNA_OPENAI_PART_LITERAL
    | LUNA_OPENAI_PART_FINISH
    | LUNA_OPENAI_PART_RETRYABLE => self.phase = LUNA_OPENAI_COMPAT_LITERAL
    LUNA_OPENAI_PART_ID_PREFIX => self.phase = LUNA_OPENAI_COMPAT_RAW
    LUNA_OPENAI_PART_MODEL | LUNA_OPENAI_PART_PAYLOAD =>
      self.phase = LUNA_OPENAI_COMPAT_LOAD_ESCAPE
    _ => {
      self.decimal_value = self.decimal_for_part()
      self.decimal_divisor = 1UL
      self.phase = LUNA_OPENAI_COMPAT_DECIMAL_PREPARE
    }
  }
}

///|
fn LunaOpenAICompatWorkspace::advance_part(
  self : LunaOpenAICompatWorkspace,
) -> Unit {
  self.part_index += 1
  self.phase = LUNA_OPENAI_COMPAT_CONFIGURE_PART
}

///|
fn LunaOpenAICompatWorkspace::progress_literal(
  self : LunaOpenAICompatWorkspace,
) -> Unit raise LunaOpenAICompatError {
  let literal = self.current_literal()
  if self.part_cursor == literal.length() {
    self.advance_part()
  } else {
    self.emit_byte(literal[self.part_cursor])
    self.part_cursor += 1
  }
}

///|
fn LunaOpenAICompatWorkspace::progress_raw(
  self : LunaOpenAICompatWorkspace,
) -> Unit raise LunaOpenAICompatError {
  if self.part_cursor == self.response_id_prefix_length {
    self.advance_part()
  } else {
    self.emit_byte(self.response_id_prefix[self.part_cursor])
    self.part_cursor += 1
  }
}

///|
fn LunaOpenAICompatWorkspace::progress_load_escape(
  self : LunaOpenAICompatWorkspace,
  event : @luna_event.LunaEventView,
) -> Unit raise LunaOpenAICompatError {
  if self.source_cursor == self.part_source_length() {
    self.advance_part()
    return
  }
  self.current_byte = if self.part_kind == LUNA_OPENAI_PART_MODEL {
    self.model_alias[self.source_cursor]
  } else {
    self.payload_byte_at(event, self.source_cursor)
  }
  let (kind, length) = luna_openai_escape_kind(self.current_byte)
  self.escape_kind = kind
  self.escape_length = length
  self.escape_cursor = 0
  self.source_cursor += 1
  self.phase = LUNA_OPENAI_COMPAT_EMIT_ESCAPE
}

///|
fn LunaOpenAICompatWorkspace::progress_emit_escape(
  self : LunaOpenAICompatWorkspace,
) -> Unit raise LunaOpenAICompatError {
  self.emit_byte(
    luna_openai_escaped_byte(
      self.escape_kind,
      self.current_byte,
      self.escape_cursor,
    ),
  )
  self.escape_cursor += 1
  if self.escape_cursor == self.escape_length {
    self.phase = LUNA_OPENAI_COMPAT_LOAD_ESCAPE
  }
}

///|
fn LunaOpenAICompatWorkspace::progress_decimal_prepare(
  self : LunaOpenAICompatWorkspace,
) -> Unit {
  if self.decimal_value / self.decimal_divisor >= 10UL {
    self.decimal_divisor *= 10UL
  } else {
    self.phase = LUNA_OPENAI_COMPAT_DECIMAL_EMIT
  }
}

///|
fn LunaOpenAICompatWorkspace::progress_decimal_emit(
  self : LunaOpenAICompatWorkspace,
) -> Unit raise LunaOpenAICompatError {
  let digit = self.decimal_value / self.decimal_divisor % 10UL
  self.emit_byte((b'0'.to_int() + digit.to_int()).to_byte())
  if self.decimal_divisor == 1UL {
    self.advance_part()
  } else {
    self.decimal_divisor /= 10UL
  }
}

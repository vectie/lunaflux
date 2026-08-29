///|
fn luna_json_hex(byte : Byte) -> Int {
  if byte >= b'0' && byte <= b'9' {
    return byte.to_int() - 48
  }
  if byte >= b'a' && byte <= b'f' {
    return byte.to_int() - 87
  }
  if byte >= b'A' && byte <= b'F' {
    return byte.to_int() - 55
  }
  -1
}

///|
fn luna_json_codepoint_width(value : Int) -> Int {
  if value <= 0x7f {
    1
  } else if value <= 0x7ff {
    2
  } else if value <= 0xffff {
    3
  } else {
    4
  }
}

///|
fn luna_json_codepoint_byte(value : Int, width : Int, index : Int) -> Byte {
  match (width, index) {
    (1, _) => value.to_byte()
    (2, 0) => (0xc0 | (value >> 6)).to_byte()
    (2, _) => (0x80 | (value & 0x3f)).to_byte()
    (3, 0) => (0xe0 | (value >> 12)).to_byte()
    (3, 1) => (0x80 | ((value >> 6) & 0x3f)).to_byte()
    (3, _) => (0x80 | (value & 0x3f)).to_byte()
    (4, 0) => (0xf0 | (value >> 18)).to_byte()
    (4, 1) => (0x80 | ((value >> 12) & 0x3f)).to_byte()
    (4, 2) => (0x80 | ((value >> 6) & 0x3f)).to_byte()
    _ => (0x80 | (value & 0x3f)).to_byte()
  }
}

///|
fn luna_known_byte(group : Int, candidate : Int, index : Int) -> Int {
  let bytes = match (group, candidate) {
    (1, 0) => b"model"
    (1, 1) => b"input"
    (1, 2) => b"stream"
    (1, 3) => b"messages"
    (1, 4) => b"temperature"
    (1, 5) => b"top_k"
    (1, 6) => b"top_p"
    (1, 7) => b"seed"
    (1, 8) => b"stop"
    (1, _) => b"stop_token_ids"
    (2, 0) => b"role"
    (2, _) => b"content"
    (3, 0) => b"system"
    (3, 1) => b"user"
    (3, _) => b"assistant"
    _ => b""
  }
  if index < bytes.length() {
    bytes[index].to_int()
  } else {
    -1
  }
}

///|
fn luna_known_length(group : Int, candidate : Int) -> Int {
  match (group, candidate) {
    (1, 0) | (1, 1) => 5
    (1, 2) => 6
    (1, 5) | (1, 6) => 5
    (1, 3) => 8
    (1, 4) => 11
    (1, 7) | (1, 8) => 4
    (1, _) => 14
    (2, 0) => 4
    (2, _) => 7
    (3, 0) => 6
    (3, 1) => 4
    (3, _) => 9
    _ => 0
  }
}

///|
fn LunaOpenAIInboundWorkspace::accept_string_byte(
  self : Self,
  byte : Byte,
) -> Unit raise LunaOpenAICompatError {
  let group = match self.string_purpose {
    LUNA_STRING_TOP_KEY => 1
    LUNA_STRING_MESSAGE_KEY => 2
    LUNA_STRING_ROLE => 3
    _ => 0
  }
  if group != 0 {
    let maximum = if group == 1 { 10 } else if group == 2 { 2 } else { 3 }
    for candidate in 0..<maximum {
      let bit = 1 << candidate
      if (self.string_candidates & bit) != 0 &&
        luna_known_byte(group, candidate, self.string_decoded_index) !=
        byte.to_int() {
        self.string_candidates -= bit
      }
    }
  } else if self.string_purpose == LUNA_STRING_MODEL {
    if self.string_decoded_index >= self.model_alias_length ||
      self.model_alias[self.string_decoded_index] != byte {
      self.string_candidates = 0
    }
  } else if self.string_purpose == LUNA_STRING_STOP {
    if self.stop_string_count >= self.inference.max_stop_strings() ||
      self.current_stop_length >= self.inference.max_stop_string_bytes() {
      raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
    }
    if self.stop_string_byte_count >= self.stop_string_bytes.length() {
      raise LunaOpenAICompatInvalid(LunaOpenAICompatCapacity)
    }
    self.stop_string_bytes[self.stop_string_byte_count] = byte
    self.stop_string_byte_count += 1
    self.current_stop_length += 1
  }
  self.string_decoded_index += 1
  if self.string_decoded_index > self.inference.max_text_bytes() {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
  }
}

///|
fn LunaOpenAIInboundWorkspace::start_string(
  self : Self,
  purpose : Int,
  return_phase : Int,
) -> Unit {
  if purpose == LUNA_STRING_STOP {
    self.current_stop_start = self.stop_string_byte_count
    self.current_stop_length = 0
  }
  self.string_purpose = purpose
  self.string_return_phase = return_phase
  self.string_start = self.cursor + 1
  self.string_decoded_index = 0
  self.string_candidates = match purpose {
    LUNA_STRING_TOP_KEY => 1023
    LUNA_STRING_MESSAGE_KEY => 3
    LUNA_STRING_ROLE => 7
    LUNA_STRING_MODEL => 1
    _ => 0
  }
  self.escape_state = 0
  self.unicode_value = 0
  self.unicode_digits = 0
  self.high_surrogate = 0
  self.pending_width = 0
  self.pending_index = 0
  self.utf8_remaining = 0
  self.cursor += 1
  self.phase = LUNA_INBOUND_STRING
}

///|
fn LunaOpenAIInboundWorkspace::advance_raw_utf8(
  self : Self,
  byte : Byte,
) -> Bool {
  let value = byte.to_int()
  if self.utf8_remaining == 0 {
    if value <= 0x7f {
      return true
    }
    if value >= 0xc2 && value <= 0xdf {
      self.utf8_remaining = 1
      self.utf8_codepoint = value & 0x1f
      self.utf8_minimum = 0x80
      return true
    }
    if value >= 0xe0 && value <= 0xef {
      self.utf8_remaining = 2
      self.utf8_codepoint = value & 0x0f
      self.utf8_minimum = 0x800
      return true
    }
    if value >= 0xf0 && value <= 0xf4 {
      self.utf8_remaining = 3
      self.utf8_codepoint = value & 7
      self.utf8_minimum = 0x10000
      return true
    }
    return false
  }
  if value < 0x80 || value > 0xbf {
    return false
  }
  self.utf8_codepoint = (self.utf8_codepoint << 6) | (value & 0x3f)
  self.utf8_remaining -= 1
  if self.utf8_remaining == 0 {
    let cp = self.utf8_codepoint
    return cp >= self.utf8_minimum &&
      cp <= 0x10ffff &&
      !(cp >= 0xd800 && cp <= 0xdfff)
  }
  true
}

///|
fn LunaOpenAIInboundWorkspace::finish_string(
  self : Self,
) -> Unit raise LunaOpenAICompatError {
  let raw_length = self.cursor - self.string_start
  match self.string_purpose {
    LUNA_STRING_TOP_KEY => {
      let mut found = -1
      for candidate in 0..<10 {
        if (self.string_candidates & (1 << candidate)) != 0 &&
          self.string_decoded_index == luna_known_length(1, candidate) {
          if found >= 0 {
            raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
          }
          found = candidate
        }
      }
      if found < 0 ||
        (self.route == 1 && found == 3) ||
        (self.route == 2 && found == 1) {
        raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
      }
      let bit = 1 << found
      if (self.top_seen & bit) != 0 {
        raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
      }
      self.top_seen = self.top_seen | bit
      self.current_key = found + 1
    }
    LUNA_STRING_MESSAGE_KEY => {
      let found = if self.string_candidates == 1 &&
        self.string_decoded_index == 4 {
        1
      } else if self.string_candidates == 2 && self.string_decoded_index == 7 {
        2
      } else {
        0
      }
      if found == 0 || (self.message_seen & (1 << (found - 1))) != 0 {
        raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
      }
      self.message_seen = self.message_seen | (1 << (found - 1))
      self.current_key = found
    }
    LUNA_STRING_MODEL =>
      if self.string_candidates == 0 ||
        self.string_decoded_index != self.model_alias_length {
        raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
      }
    LUNA_STRING_INPUT => {
      if self.string_decoded_index == 0 {
        raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
      }
      self.input_start = self.string_start
      self.input_length = raw_length
    }
    LUNA_STRING_ROLE => {
      let mut role = 0
      for candidate in 0..<3 {
        if self.string_candidates == 1 << candidate &&
          self.string_decoded_index == luna_known_length(3, candidate) {
          role = candidate + 1
        }
      }
      if role == 0 {
        raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
      }
      self.current_role = role
    }
    LUNA_STRING_CONTENT => {
      if self.string_decoded_index == 0 {
        raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
      }
      self.current_content_start = self.string_start
      self.current_content_length = raw_length
    }
    LUNA_STRING_STOP => {
      if self.current_stop_length <= 0 ||
        self.stop_string_count >= self.inference.max_stop_strings() {
        raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
      }
      let base = self.stop_string_count * 2
      self.stop_string_cells[base] = self.current_stop_start
      self.stop_string_cells[base + 1] = self.current_stop_length
      self.stop_string_count += 1
    }
    _ => abort("Luna OpenAI string purpose invariant")
  }
  self.cursor += 1
  self.phase = self.string_return_phase
}

///|
fn LunaOpenAIInboundWorkspace::progress_string_emit(
  self : Self,
) -> Unit raise LunaOpenAICompatError {
  let byte = luna_json_codepoint_byte(
    self.pending_codepoint,
    self.pending_width,
    self.pending_index,
  )
  self.accept_string_byte(byte)
  self.pending_index += 1
  if self.pending_index == self.pending_width {
    self.pending_width = 0
    self.pending_index = 0
    self.phase = LUNA_INBOUND_STRING
  }
}

///|
fn LunaOpenAIInboundWorkspace::progress_string(
  self : Self,
) -> Unit raise LunaOpenAICompatError {
  if self.cursor >= self.body_length {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
  }
  let byte = self.body_byte(self.cursor)
  if self.escape_state == 0 {
    if byte == b'"' {
      if self.utf8_remaining != 0 || self.high_surrogate != 0 {
        raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
      }
      self.finish_string()
      return
    }
    if byte == b'\\' {
      if self.utf8_remaining != 0 {
        raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
      }
      self.escape_state = 1
      self.cursor += 1
      return
    }
    if byte < 0x20 || !self.advance_raw_utf8(byte) {
      raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
    }
    self.accept_string_byte(byte)
    self.cursor += 1
    return
  }
  if self.escape_state == 1 {
    if byte == b'u' {
      self.escape_state = 2
      self.unicode_value = 0
      self.unicode_digits = 0
      self.cursor += 1
      return
    }
    let cp = match byte {
      b'"' => 34
      b'\\' => 92
      b'/' => 47
      b'b' => 8
      b'f' => 12
      b'n' => 10
      b'r' => 13
      b't' => 9
      _ => -1
    }
    if cp < 0 {
      raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
    }
    self.cursor += 1
    self.escape_state = 0
    self.pending_codepoint = cp
    self.pending_width = 1
    self.pending_index = 0
    self.phase = LUNA_INBOUND_STRING_EMIT
    return
  }
  if self.escape_state == 3 {
    if byte != b'\\' {
      raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
    }
    self.cursor += 1
    self.escape_state = 4
    return
  }
  if self.escape_state == 4 {
    if byte != b'u' {
      raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
    }
    self.cursor += 1
    self.escape_state = 5
    self.unicode_value = 0
    self.unicode_digits = 0
    return
  }
  let hex = luna_json_hex(byte)
  if hex < 0 {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
  }
  self.unicode_value = (self.unicode_value << 4) | hex
  self.unicode_digits += 1
  self.cursor += 1
  if self.unicode_digits < 4 {
    return
  }
  if self.escape_state == 2 &&
    self.unicode_value >= 0xd800 &&
    self.unicode_value <= 0xdbff {
    self.high_surrogate = self.unicode_value
    self.escape_state = 3
    return
  }
  let cp = if self.escape_state == 5 {
    if self.unicode_value < 0xdc00 || self.unicode_value > 0xdfff {
      raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
    }
    0x10000 +
    ((self.high_surrogate - 0xd800) << 10) +
    (self.unicode_value - 0xdc00)
  } else {
    if self.unicode_value >= 0xd800 && self.unicode_value <= 0xdfff {
      raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
    }
    self.unicode_value
  }
  self.high_surrogate = 0
  self.escape_state = 0
  self.pending_codepoint = cp
  self.pending_width = luna_json_codepoint_width(cp)
  self.pending_index = 0
  self.phase = LUNA_INBOUND_STRING_EMIT
}

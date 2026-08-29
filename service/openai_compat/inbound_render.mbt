///|
fn LunaOpenAIInboundWorkspace::emit_prompt_byte(
  self : Self,
  byte : Byte,
) -> Unit raise LunaOpenAICompatError {
  if self.prompt_length >= self.inference.max_text_bytes() ||
    (
      self.route == 2 &&
      self.prompt_length >= self.template.max_rendered_prompt_bytes
    ) {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatCapacity)
  }
  if self.handoff_writes.length() != 1 {
    abort("Luna OpenAI typed handoff write invariant")
  }
  self.handoff_writes[0].push_prompt_byte(byte) catch {
    _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
  }
  self.prompt_length += 1
}

///|
fn LunaOpenAIInboundWorkspace::start_render_source(
  self : Self,
  start : Int,
  length : Int,
) -> Unit {
  self.render_source_start = start
  self.render_source_length = length
  self.render_source_index = 0
  self.escape_state = 0
  self.unicode_value = 0
  self.unicode_digits = 0
  self.high_surrogate = 0
  self.pending_width = 0
  self.pending_index = 0
  self.utf8_remaining = 0
  self.phase = LUNA_INBOUND_RENDER_STRING
}

///|
fn LunaOpenAIInboundWorkspace::finish_render_source(
  self : Self,
) -> Unit raise LunaOpenAICompatError {
  if self.escape_state != 0 ||
    self.pending_width != 0 ||
    self.utf8_remaining != 0 ||
    self.high_surrogate != 0 {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
  }
  if self.route == 1 {
    self.phase = LUNA_INBOUND_HANDOFF_BEGIN
    return
  }
  let role = self.message_cells[self.render_message * 3]
  self.render_segment = match role {
    1 => 1
    2 => 3
    _ => 5
  }
  self.render_segment_index = 0
  self.phase = LUNA_INBOUND_RENDER_SEGMENT
}

///|
fn LunaOpenAIInboundWorkspace::progress_render_emit(
  self : Self,
) -> Unit raise LunaOpenAICompatError {
  self.emit_prompt_byte(
    luna_json_codepoint_byte(
      self.pending_codepoint,
      self.pending_width,
      self.pending_index,
    ),
  )
  self.pending_index += 1
  if self.pending_index == self.pending_width {
    self.pending_index = 0
    self.pending_width = 0
    self.phase = LUNA_INBOUND_RENDER_STRING
  }
}

///|
fn LunaOpenAIInboundWorkspace::progress_render_string(
  self : Self,
) -> Unit raise LunaOpenAICompatError {
  if self.render_source_index == self.render_source_length {
    self.finish_render_source()
    return
  }
  let byte = self.body_byte(self.render_source_start + self.render_source_index)
  if self.escape_state == 0 {
    if byte == b'\\' {
      if self.utf8_remaining != 0 {
        raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
      }
      self.escape_state = 1
      self.render_source_index += 1
      return
    }
    if byte < 0x20 || !self.advance_raw_utf8(byte) {
      raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
    }
    self.emit_prompt_byte(byte)
    self.render_source_index += 1
    return
  }
  if self.escape_state == 1 {
    if byte == b'u' {
      self.escape_state = 2
      self.unicode_value = 0
      self.unicode_digits = 0
      self.render_source_index += 1
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
    self.render_source_index += 1
    self.escape_state = 0
    self.pending_codepoint = cp
    self.pending_width = 1
    self.pending_index = 0
    self.phase = LUNA_INBOUND_RENDER_EMIT
    return
  }
  if self.escape_state == 3 {
    if byte != b'\\' {
      raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
    }
    self.render_source_index += 1
    self.escape_state = 4
    return
  }
  if self.escape_state == 4 {
    if byte != b'u' {
      raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
    }
    self.render_source_index += 1
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
  self.render_source_index += 1
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
  self.phase = LUNA_INBOUND_RENDER_EMIT
}

///|
fn LunaOpenAIInboundWorkspace::progress_render_segment(
  self : Self,
) -> Unit raise LunaOpenAICompatError {
  let length = self.template.segment_length(self.render_segment)
  if self.render_segment_index < length {
    self.emit_prompt_byte(
      self.template.segment_byte_at(
        self.render_segment,
        self.render_segment_index,
      ),
    )
    self.render_segment_index += 1
    return
  }
  if self.render_segment == 0 ||
    self.render_segment == 2 ||
    self.render_segment == 4 {
    let base = self.render_message * 3
    self.start_render_source(
      self.message_cells[base + 1],
      self.message_cells[base + 2],
    )
  } else if self.render_segment == 6 {
    if self.prompt_length <= 0 ||
      self.prompt_length > self.template.max_rendered_prompt_bytes {
      raise LunaOpenAICompatInvalid(LunaOpenAICompatCapacity)
    }
    self.phase = LUNA_INBOUND_HANDOFF_BEGIN
  } else {
    self.render_message += 1
    if self.render_message < self.message_count {
      let role = self.message_cells[self.render_message * 3]
      self.render_segment = match role {
        1 => 0
        2 => 2
        _ => 4
      }
    } else {
      self.render_segment = 6
    }
    self.render_segment_index = 0
  }
}

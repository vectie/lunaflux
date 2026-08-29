///|
fn luna_json_space(byte : Byte) -> Bool {
  byte == b' ' || byte == b'\t' || byte == b'\r' || byte == b'\n'
}

///|
fn LunaOpenAIInboundWorkspace::required_top_bits(self : Self) -> Int {
  if self.route == 1 {
    7
  } else {
    13
  }
}

///|
fn LunaOpenAIInboundWorkspace::finish_top(
  self : Self,
) -> Unit raise LunaOpenAICompatError {
  if (self.top_seen & self.required_top_bits()) != self.required_top_bits() {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
  }
  if self.route == 2 {
    if self.message_count == 0 ||
      self.message_cells[(self.message_count - 1) * 3] != 2 {
      raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
    }
  }
  self.phase = LUNA_INBOUND_PARSE_FINAL
}

///|
fn LunaOpenAIInboundWorkspace::finish_message(
  self : Self,
) -> Unit raise LunaOpenAICompatError {
  if self.message_seen != 3 ||
    self.current_role == 0 ||
    self.current_content_length <= 0 {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
  }
  let prior_role = if self.message_count == 0 {
    0
  } else {
    self.message_cells[(self.message_count - 1) * 3]
  }
  let valid = if self.message_count == 0 {
    self.current_role == 1 || self.current_role == 2
  } else if prior_role == 1 {
    self.current_role == 2
  } else if prior_role == 2 {
    self.current_role == 3
  } else {
    self.current_role == 2
  }
  if !valid || (self.current_role == 1 && self.message_count != 0) {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
  }
  let base = self.message_count * 3
  self.message_cells[base] = self.current_role
  self.message_cells[base + 1] = self.current_content_start
  self.message_cells[base + 2] = self.current_content_length
  self.message_count += 1
}

///|
fn LunaOpenAIInboundWorkspace::progress_true(
  self : Self,
) -> Unit raise LunaOpenAICompatError {
  let literal = b"true"
  if self.cursor >= self.body_length ||
    self.body_byte(self.cursor) != literal[self.true_index] {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
  }
  self.cursor += 1
  self.true_index += 1
  if self.true_index == 4 {
    self.phase = LUNA_INBOUND_TOP_AFTER
  }
}

///|
fn LunaOpenAIInboundWorkspace::progress_parse(
  self : Self,
) -> Unit raise LunaOpenAICompatError {
  if self.phase == LUNA_INBOUND_PARSE_FINAL {
    if self.cursor < self.body_length {
      if !luna_json_space(self.body_byte(self.cursor)) {
        raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
      }
      self.cursor += 1
      return
    }
    self.begin_render()
    return
  }
  if self.cursor >= self.body_length {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
  }
  let byte = self.body_byte(self.cursor)
  match self.phase {
    LUNA_INBOUND_PARSE_START => {
      if luna_json_space(byte) {
        self.cursor += 1
        return
      }
      if byte != b'{' {
        raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
      }
      self.cursor += 1
      self.phase = LUNA_INBOUND_TOP_KEY
    }
    LUNA_INBOUND_TOP_KEY => {
      if luna_json_space(byte) {
        self.cursor += 1
        return
      }
      if byte == b'}' {
        if self.after_comma {
          raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
        }
        self.cursor += 1
        self.finish_top()
        return
      }
      if byte != b'"' {
        raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
      }
      self.after_comma = false
      self.start_string(LUNA_STRING_TOP_KEY, LUNA_INBOUND_TOP_COLON)
    }
    LUNA_INBOUND_TOP_COLON => {
      if luna_json_space(byte) {
        self.cursor += 1
        return
      }
      if byte != b':' {
        raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
      }
      self.cursor += 1
      self.phase = LUNA_INBOUND_TOP_VALUE
    }
    LUNA_INBOUND_TOP_VALUE => {
      if luna_json_space(byte) {
        self.cursor += 1
        return
      }
      match self.current_key {
        1 => {
          if byte != b'"' {
            raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
          }
          self.start_string(LUNA_STRING_MODEL, LUNA_INBOUND_TOP_AFTER)
        }
        2 => {
          if byte != b'"' {
            raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
          }
          self.start_string(LUNA_STRING_INPUT, LUNA_INBOUND_TOP_AFTER)
        }
        3 => {
          self.true_index = 0
          self.phase = LUNA_INBOUND_TRUE
        }
        4 => self.phase = LUNA_INBOUND_MESSAGES_START
        5 | 6 | 7 | 8 =>
          self.start_number(self.current_key, LUNA_INBOUND_TOP_AFTER)
        9 =>
          if byte == b'"' {
            self.start_string(LUNA_STRING_STOP, LUNA_INBOUND_TOP_AFTER)
          } else if byte == b'[' {
            self.phase = LUNA_INBOUND_STOP_ARRAY_START
          } else {
            raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
          }
        10 => {
          if byte != b'[' {
            raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
          }
          self.phase = LUNA_INBOUND_TOKEN_ARRAY_START
        }
        _ => abort("Luna OpenAI top key invariant")
      }
    }
    LUNA_INBOUND_TOP_AFTER => {
      if luna_json_space(byte) {
        self.cursor += 1
        return
      }
      if byte == b',' {
        self.cursor += 1
        self.after_comma = true
        self.phase = LUNA_INBOUND_TOP_KEY
      } else if byte == b'}' {
        self.cursor += 1
        self.finish_top()
      } else {
        raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
      }
    }
    LUNA_INBOUND_MESSAGES_START => {
      if luna_json_space(byte) {
        self.cursor += 1
        return
      }
      if byte != b'[' {
        raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
      }
      self.cursor += 1
      self.after_comma = false
      self.phase = LUNA_INBOUND_MESSAGES_ITEM
    }
    LUNA_INBOUND_MESSAGES_ITEM => {
      if luna_json_space(byte) {
        self.cursor += 1
        return
      }
      if byte == b']' {
        if self.after_comma || self.message_count == 0 {
          raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
        }
        self.cursor += 1
        self.phase = LUNA_INBOUND_TOP_AFTER
        return
      }
      if byte != b'{' || self.message_count >= self.limits.max_messages {
        raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
      }
      self.cursor += 1
      self.after_comma = false
      self.message_seen = 0
      self.current_role = 0
      self.current_content_start = 0
      self.current_content_length = 0
      self.phase = LUNA_INBOUND_MESSAGE_KEY
    }
    LUNA_INBOUND_MESSAGE_KEY => {
      if luna_json_space(byte) {
        self.cursor += 1
        return
      }
      if byte != b'"' {
        raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
      }
      self.start_string(LUNA_STRING_MESSAGE_KEY, LUNA_INBOUND_MESSAGE_COLON)
    }
    LUNA_INBOUND_MESSAGE_COLON => {
      if luna_json_space(byte) {
        self.cursor += 1
        return
      }
      if byte != b':' {
        raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
      }
      self.cursor += 1
      self.phase = LUNA_INBOUND_MESSAGE_VALUE
    }
    LUNA_INBOUND_MESSAGE_VALUE => {
      if luna_json_space(byte) {
        self.cursor += 1
        return
      }
      if byte != b'"' {
        raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
      }
      self.start_string(
        if self.current_key == 1 {
          LUNA_STRING_ROLE
        } else {
          LUNA_STRING_CONTENT
        },
        LUNA_INBOUND_MESSAGE_AFTER,
      )
    }
    LUNA_INBOUND_MESSAGE_AFTER => {
      if luna_json_space(byte) {
        self.cursor += 1
        return
      }
      if byte == b',' {
        self.cursor += 1
        self.phase = LUNA_INBOUND_MESSAGE_KEY
      } else if byte == b'}' {
        self.cursor += 1
        self.finish_message()
        self.phase = LUNA_INBOUND_MESSAGES_AFTER
      } else {
        raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
      }
    }
    LUNA_INBOUND_MESSAGES_AFTER => {
      if luna_json_space(byte) {
        self.cursor += 1
        return
      }
      if byte == b',' {
        self.cursor += 1
        self.after_comma = true
        self.phase = LUNA_INBOUND_MESSAGES_ITEM
      } else if byte == b']' {
        self.cursor += 1
        self.phase = LUNA_INBOUND_TOP_AFTER
      } else {
        raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
      }
    }
    LUNA_INBOUND_STOP_ARRAY_START => {
      if byte != b'[' {
        raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
      }
      self.cursor += 1
      self.after_comma = false
      self.phase = LUNA_INBOUND_STOP_ARRAY_ITEM
    }
    LUNA_INBOUND_STOP_ARRAY_ITEM => {
      if luna_json_space(byte) {
        self.cursor += 1
        return
      }
      if byte == b']' {
        if self.after_comma {
          raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
        }
        self.cursor += 1
        self.phase = LUNA_INBOUND_TOP_AFTER
      } else if byte == b'"' {
        self.after_comma = false
        self.start_string(LUNA_STRING_STOP, LUNA_INBOUND_STOP_ARRAY_AFTER)
      } else {
        raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
      }
    }
    LUNA_INBOUND_STOP_ARRAY_AFTER => {
      if luna_json_space(byte) {
        self.cursor += 1
        return
      }
      if byte == b',' {
        self.cursor += 1
        self.after_comma = true
        self.phase = LUNA_INBOUND_STOP_ARRAY_ITEM
      } else if byte == b']' {
        self.cursor += 1
        self.phase = LUNA_INBOUND_TOP_AFTER
      } else {
        raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
      }
    }
    LUNA_INBOUND_TOKEN_ARRAY_START => {
      if byte != b'[' {
        raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
      }
      self.cursor += 1
      self.after_comma = false
      self.phase = LUNA_INBOUND_TOKEN_ARRAY_ITEM
    }
    LUNA_INBOUND_TOKEN_ARRAY_ITEM => {
      if luna_json_space(byte) {
        self.cursor += 1
        return
      }
      if byte == b']' {
        if self.after_comma {
          raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
        }
        self.cursor += 1
        self.phase = LUNA_INBOUND_TOP_AFTER
      } else {
        self.after_comma = false
        self.start_number(10, LUNA_INBOUND_TOKEN_ARRAY_AFTER)
      }
    }
    LUNA_INBOUND_TOKEN_ARRAY_AFTER => {
      if luna_json_space(byte) {
        self.cursor += 1
        return
      }
      if byte == b',' {
        self.cursor += 1
        self.after_comma = true
        self.phase = LUNA_INBOUND_TOKEN_ARRAY_ITEM
      } else if byte == b']' {
        self.cursor += 1
        self.phase = LUNA_INBOUND_TOP_AFTER
      } else {
        raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
      }
    }
    _ => abort("Luna OpenAI parser phase invariant")
  }
}

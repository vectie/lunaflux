///|
fn luna_json_number_delimiter(byte : Byte) -> Bool {
  luna_json_space(byte) || byte == b',' || byte == b'}' || byte == b']'
}

///|
fn LunaOpenAIInboundWorkspace::start_number(
  self : Self,
  target : Int,
  return_phase : Int,
) -> Unit {
  self.number_target = target
  self.number_return_phase = return_phase
  self.number_stage = 0
  self.number_negative = false
  self.number_significand = 0UL
  self.number_fraction_digits = 0
  self.number_exponent_negative = false
  self.number_exponent = 0
  self.number_leading_zero = false
  self.number_value = 0.0
  self.number_scale = 0
  self.phase = LUNA_INBOUND_NUMBER
}

///|
fn LunaOpenAIInboundWorkspace::add_significand_digit(
  self : Self,
  digit : Int,
) -> Unit raise LunaOpenAICompatError {
  let value = digit.to_uint64()
  if self.number_significand > (0xffffffffffffffffUL - value) / 10UL {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
  }
  self.number_significand = self.number_significand * 10UL + value
}

///|
fn LunaOpenAIInboundWorkspace::finish_number_scan(
  self : Self,
) -> Unit raise LunaOpenAICompatError {
  if self.number_stage != 1 && self.number_stage != 3 && self.number_stage != 5 {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
  }
  if self.number_target == 5 || self.number_target == 7 {
    let signed_exponent = if self.number_exponent_negative {
      -self.number_exponent
    } else {
      self.number_exponent
    }
    self.number_scale = signed_exponent - self.number_fraction_digits
    if self.number_scale < -400 || self.number_scale > 400 {
      raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
    }
    self.number_value = self.number_significand.to_double()
    if self.number_negative {
      self.number_value = -self.number_value
    }
    self.phase = LUNA_INBOUND_NUMBER_SCALE
  } else {
    if self.number_negative ||
      self.number_fraction_digits != 0 ||
      self.number_stage == 5 {
      raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
    }
    self.phase = LUNA_INBOUND_NUMBER_FINAL
  }
}

///|
fn LunaOpenAIInboundWorkspace::progress_number(
  self : Self,
) -> Unit raise LunaOpenAICompatError {
  if self.cursor >= self.body_length {
    self.finish_number_scan()
    return
  }
  let byte = self.body_byte(self.cursor)
  if luna_json_number_delimiter(byte) {
    self.finish_number_scan()
    return
  }
  if self.number_stage == 0 && byte == b'-' {
    self.number_negative = true
    self.number_stage = 6
    self.cursor += 1
    return
  }
  let digit = if byte >= b'0' && byte <= b'9' { byte.to_int() - 48 } else { -1 }
  if self.number_stage == 0 || self.number_stage == 6 {
    if digit < 0 {
      raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
    }
    self.number_leading_zero = digit == 0
    self.add_significand_digit(digit)
    self.number_stage = 1
    self.cursor += 1
    return
  }
  if self.number_stage == 1 {
    if digit >= 0 {
      if self.number_leading_zero {
        raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
      }
      self.add_significand_digit(digit)
      self.cursor += 1
      return
    }
    if byte == b'.' {
      self.number_stage = 2
      self.cursor += 1
      return
    }
    if byte == b'e' || byte == b'E' {
      self.number_stage = 4
      self.cursor += 1
      return
    }
    raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
  }
  if self.number_stage == 2 || self.number_stage == 3 {
    if digit >= 0 {
      self.add_significand_digit(digit)
      self.number_fraction_digits += 1
      self.number_stage = 3
      self.cursor += 1
      return
    }
    if self.number_stage == 3 && (byte == b'e' || byte == b'E') {
      self.number_stage = 4
      self.cursor += 1
      return
    }
    raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
  }
  if self.number_stage == 4 {
    if byte == b'+' || byte == b'-' {
      self.number_exponent_negative = byte == b'-'
      self.number_stage = 7
      self.cursor += 1
      return
    }
  }
  if self.number_stage == 4 || self.number_stage == 5 || self.number_stage == 7 {
    if digit < 0 || self.number_exponent > 400 {
      raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
    }
    self.number_exponent = self.number_exponent * 10 + digit
    self.number_stage = 5
    self.cursor += 1
    return
  }
  raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
}

///|
fn LunaOpenAIInboundWorkspace::progress_number_scale(self : Self) -> Unit {
  if self.number_scale < 0 {
    self.number_value /= 10.0
    self.number_scale += 1
  } else if self.number_scale > 0 {
    self.number_value *= 10.0
    self.number_scale -= 1
  } else {
    self.phase = LUNA_INBOUND_NUMBER_FINAL
  }
}

///|
fn LunaOpenAIInboundWorkspace::progress_number_final(
  self : Self,
) -> Unit raise LunaOpenAICompatError {
  match self.number_target {
    5 => {
      self.sampling_mode = 1
      self.temperature = self.number_value
    }
    6 => {
      self.sampling_mode = 1
      self.has_top_k = true
      self.top_k = self.number_significand.to_int()
    }
    7 => {
      self.sampling_mode = 1
      self.has_top_p = true
      self.top_p = self.number_value
    }
    8 => self.sampling_seed_value = self.number_significand
    10 => {
      if self.stop_token_count >= self.inference.max_stop_token_ids() {
        raise LunaOpenAICompatInvalid(LunaOpenAICompatJson)
      }
      self.stop_tokens[self.stop_token_count] = self.number_significand.to_int()
      self.stop_token_count += 1
    }
    _ => abort("Luna OpenAI number target invariant")
  }
  self.phase = self.number_return_phase
}

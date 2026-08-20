///|
fn luna_request_safe_identifier_byte(byte : Byte) -> Bool {
  (byte >= b'a' && byte <= b'z') ||
  (byte >= b'A' && byte <= b'Z') ||
  (byte >= b'0' && byte <= b'9') ||
  byte == b'-' ||
  byte == b'_' ||
  byte == b'.' ||
  byte == b':'
}

///|
fn luna_request_lower_hex_byte(byte : Byte) -> Bool {
  (byte >= b'0' && byte <= b'9') || (byte >= b'a' && byte <= b'f')
}

///|
fn LunaFramedRequestWorkspace::sampling_header_status(
  self : LunaFramedRequestWorkspace,
) -> Int {
  let mode = read_i32(self.storage, 60)
  let top_k = read_i32(self.storage, 64)
  let temperature = read_u64(self.storage, 72).reinterpret_as_double()
  let top_p_bits = read_u64(self.storage, 80)
  if (self.flags & 2) == 0 {
    return if mode == 0 &&
      (self.flags & 12) == 0 &&
      top_k == 0 &&
      read_u64(self.storage, 72) == 1.0.reinterpret_as_uint64() &&
      top_p_bits == 0UL {
      0
    } else {
      1
    }
  }
  if mode != 1 ||
    ((self.flags & 4) != 0) != (top_k != 0) ||
    ((self.flags & 8) != 0) != (top_p_bits != 0UL) {
    return 1
  }
  if temperature.is_nan() ||
    temperature.is_inf() ||
    temperature <= 0.0 ||
    temperature > self.limits.inference_limits().max_temperature() {
    return 2
  }
  if (self.flags & 4) != 0 &&
    (top_k <= 0 || top_k > self.limits.inference_limits().max_top_k()) {
    return 2
  }
  if (self.flags & 8) != 0 {
    let top_p = top_p_bits.reinterpret_as_double()
    if top_p.is_nan() || top_p.is_inf() || top_p <= 0.0 || top_p > 1.0 {
      return 2
    }
  }
  0
}

///|
fn LunaFramedRequestWorkspace::reset_utf8(
  self : LunaFramedRequestWorkspace,
) -> Unit {
  self.utf8_remaining = 0
  self.utf8_codepoint = 0
  self.utf8_minimum = 0
}

///|
fn LunaFramedRequestWorkspace::advance_utf8(
  self : LunaFramedRequestWorkspace,
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
      self.utf8_codepoint = value & 0x07
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
    let codepoint = self.utf8_codepoint
    if codepoint < self.utf8_minimum ||
      codepoint > 0x10ffff ||
      (codepoint >= 0xd800 && codepoint <= 0xdfff) {
      return false
    }
  }
  true
}

///|
#valtype
priv struct Utf8AdvanceStatus(Bool, Int, Int, Int, Int)

///|
fn advance_utf8_status(
  remaining : Int,
  codepoint : Int,
  minimum : Int,
  seen : Int,
  byte : Byte,
) -> Utf8AdvanceStatus {
  let value = byte.to_int()
  if remaining == 0 {
    if value <= 0x7f {
      return Utf8AdvanceStatus(true, 0, 0, 0, 0)
    }
    if value >= 0xc2 && value <= 0xdf {
      return Utf8AdvanceStatus(true, 1, value & 0x1f, 0x80, 1)
    }
    if value >= 0xe0 && value <= 0xef {
      return Utf8AdvanceStatus(true, 2, value & 0x0f, 0x800, 1)
    }
    if value >= 0xf0 && value <= 0xf4 {
      return Utf8AdvanceStatus(true, 3, value & 0x07, 0x10000, 1)
    }
    return Utf8AdvanceStatus(false, 0, 0, 0, 0)
  }
  if value < 0x80 || value > 0xbf {
    return Utf8AdvanceStatus(false, 0, 0, 0, 0)
  }
  let next_remaining = remaining - 1
  let next_codepoint = (codepoint << 6) | (value & 0x3f)
  let next_seen = seen + 1
  if next_remaining == 0 {
    if next_codepoint < minimum ||
      next_codepoint > 0x10ffff ||
      (next_codepoint >= 0xd800 && next_codepoint <= 0xdfff) {
      return Utf8AdvanceStatus(false, 0, 0, 0, 0)
    }
    Utf8AdvanceStatus(true, 0, 0, 0, 0)
  } else {
    Utf8AdvanceStatus(true, next_remaining, next_codepoint, minimum, next_seen)
  }
}

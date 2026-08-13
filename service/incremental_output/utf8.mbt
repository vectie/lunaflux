///|
#valtype
priv struct Utf8Advance(Int, Int, Int, Int)

///|
fn advance_utf8(
  remaining : Int,
  codepoint : Int,
  minimum : Int,
  seen : Int,
  byte : Byte,
) -> Utf8Advance raise IncrementalOutputError {
  let value = byte.to_int()
  if remaining == 0 {
    if value <= 0x7f {
      return Utf8Advance(0, 0, 0, 0)
    }
    if value >= 0xc2 && value <= 0xdf {
      return Utf8Advance(1, value & 0x1f, 0x80, 1)
    }
    if value >= 0xe0 && value <= 0xef {
      return Utf8Advance(2, value & 0x0f, 0x800, 1)
    }
    if value >= 0xf0 && value <= 0xf4 {
      return Utf8Advance(3, value & 0x07, 0x10000, 1)
    }
    raise invalid(Utf8, Invalid)
  }
  if value < 0x80 || value > 0xbf {
    raise invalid(Utf8, Invalid)
  }
  let next_remaining = remaining - 1
  let next_codepoint = (codepoint << 6) | (value & 0x3f)
  let next_seen = seen + 1
  if next_remaining == 0 {
    if next_codepoint < minimum ||
      next_codepoint > 0x10ffff ||
      (next_codepoint >= 0xd800 && next_codepoint <= 0xdfff) {
      raise invalid(Utf8, Invalid)
    }
    Utf8Advance(0, 0, 0, 0)
  } else {
    Utf8Advance(next_remaining, next_codepoint, minimum, next_seen)
  }
}

///|
/// Python's .1f rounding uses the exact binary Double, not a rounded x*10.
/// Multiply its integer significand by five, then round the remaining power
/// of two denominator ties-to-even. This preserves e.g. 0.15 -> "0.1".
fn timestamp(value : Double) -> String raise PromptError {
  guard !value.is_nan() &&
    !value.is_inf() &&
    value >= 0.0 &&
    value <= 214748364.0 else {
    raise InvalidMedia
  }
  let bits = value.reinterpret_as_uint64()
  let exponent = ((bits >> 52) & 0x7ffUL).to_int()
  let mantissa = (bits & 0xfffffffffffffUL) +
    (if exponent == 0 { 0UL } else { 0x10000000000000UL })
  let numerator = mantissa * 5UL
  let shift = if exponent == 0 { 1073 } else { 1074 - exponent }
  let rounded = if shift >= 64 {
    0UL
  } else {
    let quotient = numerator >> shift
    let remainder = numerator & ((1UL << shift) - 1UL)
    let half = 1UL << (shift - 1)
    quotient +
    (if remainder > half || (remainder == half && quotient % 2UL == 1UL) {
      1UL
    } else {
      0UL
    })
  }
  let sign = if bits >> 63 == 1UL { "-" } else { "" }
  "<\{sign}\{rounded/10UL}.\{rounded%10UL} seconds>"
}

///|
fn pow2(exponent : Int) -> Float {
  if exponent >= -126 {
    Float::reinterpret_from_uint((exponent + 127).reinterpret_as_uint() << 23)
  } else {
    Float::reinterpret_from_uint(1U << (exponent + 149))
  }
}

///|
/// Exact finite storage decode; E5M2 NaN/Inf and E4M3FN NaN are rejected.
pub fn decode_element(
  element : Element,
  code : UInt,
) -> Float raise PrecisionError {
  let bits = element.bits()
  if bits < 32 && code >= 1U << bits {
    raise InvalidCode(code.reinterpret_as_int())
  }
  match element {
    F32 => {
      let value = Float::reinterpret_from_uint(code)
      if value.is_nan() || value.is_inf() {
        raise NonFiniteValue
      }
      value
    }
    Bf16 => decode_element(F32, code << 16)
    Signed8 =>
      Float::from_int(
        if code >= 128U {
          code.reinterpret_as_int() - 256
        } else {
          code.reinterpret_as_int()
        },
      )
    Signed4 =>
      Float::from_int(
        if code >= 8U {
          code.reinterpret_as_int() - 16
        } else {
          code.reinterpret_as_int()
        },
      )
    E2M1 => {
      let magnitude : Float = match code & 7U {
        0U => 0.0F
        1U => 0.5F
        2U => 1.0F
        3U => 1.5F
        4U => 2.0F
        5U => 3.0F
        6U => 4.0F
        _ => 6.0F
      }
      if (code & 8U) != 0U {
        -magnitude
      } else {
        magnitude
      }
    }
    Fp16 | E4M3 | E5M2 => {
      let (mantissa_bits, exponent_bits, bias) = match element {
        Fp16 => (10, 5, 15)
        E4M3 => (3, 4, 7)
        _ => (2, 5, 15)
      }
      let mantissa_mask = (1U << mantissa_bits) - 1U
      let exponent_mask = (1U << exponent_bits) - 1U
      let mantissa = (code & mantissa_mask).reinterpret_as_int()
      let exponent = ((code >> mantissa_bits) & exponent_mask).reinterpret_as_int()
      if (element == E4M3 && exponent == 15 && mantissa == 7) ||
        (element != E4M3 && exponent == 31) {
        raise InvalidCode(code.reinterpret_as_int())
      }
      let magnitude = if exponent == 0 {
        Float::from_int(mantissa) * pow2(1 - bias - mantissa_bits)
      } else {
        (1.0F + Float::from_int(mantissa) / Float::from_int(1 << mantissa_bits)) *
        pow2(exponent - bias)
      }
      if (code & (1U << (bits - 1))) != 0U {
        -magnitude
      } else {
        magnitude
      }
    }
  }
}

///|
pub fn decode_scale(
  element : ScaleElement,
  code : UInt,
) -> Float raise PrecisionError {
  let value = match element {
    Absent => raise InvalidScale
    F32 => decode_element(Element::F32, code)
    E4M3 => decode_element(Element::E4M3, code)
    Ue8M0 => {
      if code >= 255U {
        raise InvalidScale
      }
      pow2(code.reinterpret_as_int() - 127)
    }
  }
  if value <= 0.0F || value.is_nan() || value.is_inf() {
    raise InvalidScale
  }
  value
}

///|
pub fn Element::maximum(self : Element) -> Float {
  match self {
    F32 => Float::reinterpret_from_uint(0x7f7fffffU)
    Bf16 => Float::reinterpret_from_uint(0x7f7f0000U)
    Fp16 => 65504.0F
    E4M3 => 448.0F
    E5M2 => 57344.0F
    Signed8 => 127.0F
    Signed4 => 7.0F
    E2M1 => 6.0F
  }
}

///|
/// Deterministic RNE with finite saturation. Small formats use the complete
/// codebook in this scalar oracle, not in the production device instruction.
pub fn encode_element(
  element : Element,
  value : Float,
) -> UInt raise PrecisionError {
  if value.is_nan() || value.is_inf() {
    raise NonFiniteValue
  }
  if element == F32 {
    return value.reinterpret_as_uint()
  }
  let negative = (value.reinterpret_as_uint() & 0x80000000U) != 0U
  let sign = if negative { 1U << (element.bits() - 1) } else { 0U }
  if element == Bf16 {
    let bounded = value.max(-element.maximum()).min(element.maximum())
    let bits = bounded.reinterpret_as_uint()
    return (bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16
  }
  if element == Signed4 || element == Signed8 {
    let maximum = element.maximum().to_int()
    let bounded = value
      .max(Float::from_int(-maximum - 1))
      .min(Float::from_int(maximum))
    let base = bounded.floor().to_int()
    let fraction = bounded - Float::from_int(base)
    let rounded = if fraction > 0.5F || (fraction == 0.5F && (base & 1) != 0) {
      base + 1
    } else {
      base
    }
    return rounded.reinterpret_as_uint() & ((1U << element.bits()) - 1U)
  }
  let maximum_code = match element {
    Fp16 => 0x7bffU
    E4M3 => 0x7eU
    E5M2 => 0x7bU
    E2M1 => 7U
    _ => 0U
  }
  let bounded = value.abs().min(element.maximum())
  // Monotone positive codebook: binary search avoids a 32K half-code scan.
  let low = for lo = 0U, hi = maximum_code {
    if lo >= hi {
      break lo
    }
    let mid = lo + (hi - lo + 1U) / 2U
    if decode_element(element, mid) <= bounded {
      continue mid, hi
    }
    continue lo, mid - 1U
  }
  if low == maximum_code {
    return sign | low
  }
  let a = decode_element(element, low)
  let b = decode_element(element, low + 1U)
  let high = bounded - a > b - bounded ||
    (bounded - a == b - bounded && (low & 1U) != 0U)
  sign | (if high { low + 1U } else { low })
}

///|
/// UE8M0 rounds scales upwards; it never interprets 255 as a finite exponent.
pub fn encode_scale(
  element : ScaleElement,
  value : Float,
) -> UInt raise PrecisionError {
  if value.is_nan() || value.is_inf() || value <= 0.0F {
    raise InvalidScale
  }
  match element {
    Absent => raise InvalidScale
    F32 => value.reinterpret_as_uint()
    E4M3 => {
      let code = encode_element(Element::E4M3, value)
      if code == 0U {
        1U
      } else {
        code
      }
    }
    Ue8M0 => {
      for code in 0U..<255U {
        if decode_scale(Ue8M0, code) >= value {
          return code
        }
      }
      raise InvalidScale
    }
  }
}

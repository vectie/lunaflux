///|
pub(all) suberror BlockFp8Ue8M0ReferenceError {
  InvalidE4M3Code(Int)
  InvalidE8M0Code(Int)
  InvalidShape
  NonFiniteInput(Int)
} derive(Debug, Eq)

///|
fn power_of_two(exponent : Int) -> Float {
  let mut value = 1.0F
  if exponent >= 0 {
    for _ in 0..<exponent {
      value *= 2.0F
    }
  } else {
    for _ in 0..<-exponent {
      value *= 0.5F
    }
  }
  value
}

///|
fn round_bf16_rne(value : Float) -> Float {
  let bits = value.reinterpret_as_uint()
  let rounded = bits + 0x7fffU + ((bits >> 16) & 1U)
  Float::reinterpret_from_uint(rounded & 0xffff0000U)
}

///|
/// Decode one finite NVIDIA E4M3FN storage code.
pub fn decode_e4m3_finite(
  code : Int,
) -> Float raise BlockFp8Ue8M0ReferenceError {
  if code < 0 || code > 255 || code == 0x7f || code == 0xff {
    raise InvalidE4M3Code(code)
  }
  let sign = if (code & 0x80) == 0 { 1.0F } else { -1.0F }
  let exponent = (code >> 3) & 0x0f
  let mantissa = code & 0x07
  if exponent == 0 {
    sign * Float::from_int(mantissa) * power_of_two(-9)
  } else {
    sign *
    (1.0F + Float::from_int(mantissa) * 0.125F) *
    power_of_two(exponent - 7)
  }
}

///|
/// Decode NVIDIA UE8M0FNU: codes 0..254 are 2^(code-127); 255 is NaN.
pub fn decode_ue8m0(code : Int) -> Float raise BlockFp8Ue8M0ReferenceError {
  if code < 0 || code >= 255 {
    raise InvalidE8M0Code(code)
  }
  power_of_two(code - 127)
}

///|
/// Round finite F32 to finite E4M3FN, ties to an even stored significand.
pub fn round_e4m3_finite_code(
  value : Float,
) -> Int raise BlockFp8Ue8M0ReferenceError {
  if value.is_nan() || value.is_inf() {
    raise NonFiniteInput(0)
  }
  let bounded = value.max(-448.0F).min(448.0F)
  if bounded == 0.0F {
    return 0
  }
  let sign = if bounded < 0.0F { 0x80 } else { 0 }
  let magnitude = bounded.abs()
  let mut best = 0
  let mut error = 1000000000.0F
  for code in 0..<0x7f {
    let candidate = try! decode_e4m3_finite(code)
    let candidate_error = (candidate - magnitude).abs()
    if candidate_error < error ||
      (candidate_error == error && (code & 1) == 0 && (best & 1) != 0) {
      best = code
      error = candidate_error
    }
  }
  sign | best
}

///|
/// Official activation scale code for one non-empty block of at most 128.
pub fn activation_scale_ue8m0_code(
  values : ArrayView[Float],
) -> Int raise BlockFp8Ue8M0ReferenceError {
  if values.length() <= 0 || values.length() > 128 {
    raise InvalidShape
  }
  let mut maximum = 0.0001F
  for index, value in values {
    if value.is_nan() || value.is_inf() {
      raise NonFiniteInput(index)
    }
    maximum = maximum.max(value.abs())
  }
  let requested = maximum * 0.002232142857142857F
  for code in 0..<255 {
    if (try! decode_ue8m0(code)) >= requested {
      return code
    }
  }
  254
}

///|
/// Ordered scalar oracle for one K-wide dot, K a positive multiple of 128.
pub fn block128_dot_bf16_boundary(
  input : ArrayView[Float],
  weight_codes : ArrayView[Int],
  weight_scale_codes : ArrayView[Int],
) -> Float raise BlockFp8Ue8M0ReferenceError {
  if input.length() <= 0 ||
    input.length() % 128 != 0 ||
    input.length() != weight_codes.length() ||
    weight_scale_codes.length() != input.length() / 128 {
    raise InvalidShape
  }
  let mut accumulator = 0.0F
  for block in 0..<weight_scale_codes.length() {
    let begin = block * 128
    let scale_code = try! activation_scale_ue8m0_code(input[begin:begin + 128])
    let activation_scale = try! decode_ue8m0(scale_code)
    let parameter_scale = try! decode_ue8m0(weight_scale_codes[block])
    let mut block_accumulator = 0.0F
    for offset in 0..<128 {
      let index = begin + offset
      let quantized = try! round_e4m3_finite_code(
        input[index] / activation_scale,
      )
      let activation = try! decode_e4m3_finite(quantized)
      let parameter = try! decode_e4m3_finite(weight_codes[index])
      block_accumulator += activation * parameter
    }
    accumulator += block_accumulator * (activation_scale * parameter_scale)
  }
  round_bf16_rne(accumulator)
}

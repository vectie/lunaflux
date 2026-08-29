///|
let fp8_probe_input : FixedArray[Float] = [
  4.0F,
  -4.0F,
  2.0F,
  -2.0F,
  1.0F,
  -0.5F,
  0.25F,
  0.0F,
  3.0F,
  -3.0F,
  1.5F,
  -1.5F,
  0.75F,
  -0.375F,
  0.25F,
  -0.125F,
]

///|
let fp8_probe_codes : FixedArray[Int] = [
  0x00, 0x7e, 0xfe, 0x20, 0x30, 0x38, 0xb8, 0x40, 0xc0, 0x48, 0xc8, 0x28, 0xa8, 0x44,
  0x34, 0xbc, 0x50, 0xd0,
]

///|
fn fp8_probe_ordered_mul(left : Float, right : Float) -> Float {
  left * right
}

///|
fn fp8_probe_ordered_add(left : Float, right : Float) -> Float {
  left + right
}

///|
fn fp8_probe_power_of_two(exponent : Int) -> Float {
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
fn fp8_probe_decode(code : Int) -> Float {
  let sign = if (code & 0x80) == 0 { 1.0F } else { -1.0F }
  let exponent = (code >> 3) & 0x0f
  let mantissa = code & 0x07
  if exponent == 0 {
    sign * Float::from_int(mantissa) * fp8_probe_power_of_two(-9)
  } else if exponent == 15 && mantissa == 7 {
    Float::reinterpret_from_uint(0x7fc00000U)
  } else {
    sign *
    (1.0F + Float::from_int(mantissa) * 0.125F) *
    fp8_probe_power_of_two(exponent - 7)
  }
}

///|
fn fp8_probe_round(value : Float) -> Float {
  let bounded = value.max(-448.0F).min(448.0F)
  if bounded == 0.0F {
    return 0.0F
  }
  let sign = if bounded < 0.0F { 0x80 } else { 0 }
  let magnitude = bounded.abs()
  let mut best_code = 0
  let mut best_error = 1000000000.0F
  for code in 0..<0x7f {
    let candidate = fp8_probe_decode(code)
    let error = (candidate - magnitude).abs()
    if error < best_error ||
      (error == best_error && (code & 1) == 0 && (best_code & 1) != 0) {
      best_code = code
      best_error = error
    }
  }
  fp8_probe_decode(sign | best_code)
}

///|
fn fp8_probe_bf16_bits(value : Float) -> Int {
  let bits = value.reinterpret_as_uint()
  ((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16).reinterpret_as_int()
}

///|
fn fp8_probe_read_bf16(bytes : FixedArray[Byte], index : Int) -> Float {
  let offset = index * 2
  let bits = bytes[offset].to_int() | (bytes[offset + 1].to_int() << 8)
  Float::reinterpret_from_uint(bits.reinterpret_as_uint() << 16)
}

///|
fn fp8_probe_read_f32(bytes : FixedArray[Byte], index : Int) -> Float {
  let offset = index * 4
  let bits = bytes[offset].to_uint() |
    (bytes[offset + 1].to_uint() << 8) |
    (bytes[offset + 2].to_uint() << 16) |
    (bytes[offset + 3].to_uint() << 24)
  Float::reinterpret_from_uint(bits)
}

///|
fn fp8_probe_write_i32(
  bytes : FixedArray[Byte],
  index : Int,
  value : Int,
) -> Unit {
  let offset = index * 4
  bytes[offset] = value.to_byte()
  bytes[offset + 1] = (value >> 8).to_byte()
  bytes[offset + 2] = (value >> 16).to_byte()
  bytes[offset + 3] = (value >> 24).to_byte()
}

///|
fn fp8_probe_write_f32(
  bytes : FixedArray[Byte],
  index : Int,
  value : Float,
) -> Unit {
  let bits = value.reinterpret_as_uint()
  let offset = index * 4
  bytes[offset] = bits.to_byte()
  bytes[offset + 1] = (bits >> 8).to_byte()
  bytes[offset + 2] = (bits >> 16).to_byte()
  bytes[offset + 3] = (bits >> 24).to_byte()
}

///|
fn fp8_probe_counts() -> FixedArray[Byte] {
  let bytes = FixedArray::make(20, b'\x00')
  for index, value in [1, 1, 2, 2, 2] {
    fp8_probe_write_i32(bytes, index, value)
  }
  bytes
}

///|
fn fp8_probe_input_bytes() -> FixedArray[Byte] {
  let bytes = FixedArray::make(32, b'\x00')
  for index, value in fp8_probe_input {
    let bits = fp8_probe_bf16_bits(value)
    bytes[index * 2] = bits.to_byte()
    bytes[index * 2 + 1] = (bits >> 8).to_byte()
  }
  bytes
}

///|
fn fp8_probe_weight_bytes(length : Int, offset : Int) -> FixedArray[Byte] {
  FixedArray::makei(length, index => {
    fp8_probe_codes[(index * 5 + offset * 3 + index / 8) %
    fp8_probe_codes.length()].to_byte()
  })
}

///|
fn fp8_probe_scale_bytes(value : Float) -> FixedArray[Byte] {
  let bytes = FixedArray::make(4, b'\x00')
  fp8_probe_write_f32(bytes, 0, value)
  bytes
}

///|
fn fp8_probe_dynamic_scale(values : FixedArray[Float]) -> Float {
  let mut maximum = 0.0F
  for value in values {
    maximum = maximum.max(value.abs())
  }
  if maximum == 0.0F {
    1.0F
  } else {
    maximum / 448.0F
  }
}

///|
fn fp8_probe_activation(value : Float, scale : Float) -> Float {
  fp8_probe_round(value / scale) * scale
}

///|
fn fp8_probe_qkv_expected() -> (FixedArray[Float], Float) {
  let scale = fp8_probe_dynamic_scale(fp8_probe_input)
  let query = fp8_probe_weight_bytes(64, 1)
  let key = fp8_probe_weight_bytes(32, 2)
  let value = fp8_probe_weight_bytes(32, 3)
  let result = FixedArray::make(32, 0.0F)
  for row in 0..<2 {
    for column in 0..<16 {
      let (weight, local_column, weight_scale) = if column < 8 {
        (query, column, 0.03125F)
      } else if column < 12 {
        (key, column - 8, 0.0625F)
      } else {
        (value, column - 12, 0.015625F)
      }
      let mut accumulator = 0.0F
      for inner in 0..<8 {
        let activation = fp8_probe_activation(
          fp8_probe_input[row * 8 + inner],
          scale,
        )
        let weight_value = fp8_probe_ordered_mul(
          fp8_probe_decode(weight[local_column * 8 + inner].to_int()),
          weight_scale,
        )
        accumulator = fp8_probe_ordered_add(
          accumulator,
          fp8_probe_ordered_mul(activation, weight_value),
        )
      }
      result[row * 16 + column] = accumulator
    }
  }
  (result, scale)
}

///|
fn fp8_probe_gated_expected() -> (FixedArray[Float], Float, Float) {
  let external_scale = fp8_probe_dynamic_scale(fp8_probe_input)
  let gate = fp8_probe_weight_bytes(128, 4)
  let up = fp8_probe_weight_bytes(128, 5)
  let down = fp8_probe_weight_bytes(128, 6)
  let intermediate = FixedArray::make(32, 0.0F)
  let mut maximum = 0.0F
  for row in 0..<2 {
    for column in 0..<16 {
      let mut gate_sum = 0.0F
      let mut up_sum = 0.0F
      for inner in 0..<8 {
        let activation = fp8_probe_activation(
          fp8_probe_input[row * 8 + inner],
          external_scale,
        )
        let gate_value = fp8_probe_ordered_mul(
          fp8_probe_decode(gate[column * 8 + inner].to_int()),
          0.00390625F,
        )
        let up_value = fp8_probe_ordered_mul(
          fp8_probe_decode(up[column * 8 + inner].to_int()),
          0.0078125F,
        )
        gate_sum = fp8_probe_ordered_add(
          gate_sum,
          fp8_probe_ordered_mul(activation, gate_value),
        )
        up_sum = fp8_probe_ordered_add(
          up_sum,
          fp8_probe_ordered_mul(activation, up_value),
        )
      }
      let combined = fp8_probe_ordered_mul(
        gate_sum / fp8_probe_ordered_add(1.0F, @math.expf(-gate_sum)),
        up_sum,
      )
      intermediate[row * 16 + column] = combined
      maximum = maximum.max(combined.abs())
    }
  }
  let post_scale = if maximum == 0.0F { 1.0F } else { maximum / 448.0F }
  let result = FixedArray::make(16, 0.0F)
  for row in 0..<2 {
    for column in 0..<8 {
      let mut accumulator = 0.0F
      for inner in 0..<16 {
        let activation = fp8_probe_activation(
          intermediate[row * 16 + inner],
          post_scale,
        )
        let weight = fp8_probe_ordered_mul(
          fp8_probe_decode(down[column * 16 + inner].to_int()),
          0.015625F,
        )
        accumulator = fp8_probe_ordered_add(
          accumulator,
          fp8_probe_ordered_mul(activation, weight),
        )
      }
      result[row * 8 + column] = accumulator
    }
  }
  (result, external_scale, post_scale)
}

///|
fn fp8_probe_validate_output(
  actual : FixedArray[Byte],
  expected : FixedArray[Float],
  label : String,
) -> Float raise {
  require_fp8_probe(actual.length() == expected.length() * 2, label + " size")
  let mut maximum = 0.0F
  for index, expected_value in expected {
    let observed = fp8_probe_read_bf16(actual, index)
    let error = (observed - expected_value).abs()
    require_fp8_probe(
      !observed.is_nan() &&
      !observed.is_inf() &&
      error <= 0.03125F + 0.015625F * expected_value.abs(),
      label + " differs from independent ordered F32 referee",
    )
    maximum = maximum.max(error)
  }
  maximum
}

///|
test "independent E4M3 referee covers signed finite boundary and row-major layout" {
  assert_eq(fp8_probe_decode(0x7e), 448.0F)
  assert_eq(fp8_probe_decode(0xfe), -448.0F)
  assert_eq(fp8_probe_round(448.0F), 448.0F)
  assert_eq(fp8_probe_round(-448.0F), -448.0F)
  let query = fp8_probe_weight_bytes(64, 1)
  assert_true(query[0] != query[7] && query[7] != query[8])
  let (simple, simple_scale) = fp8_probe_qkv_expected()
  let (gated, gated_scale, post_scale) = fp8_probe_gated_expected()
  assert_eq(simple.length(), 32)
  assert_eq(gated.length(), 16)
  assert_true(simple_scale > 0.0F && gated_scale > 0.0F && post_scale > 0.0F)
  assert_true(simple.any(value => value != 0.0F))
  assert_true(gated.any(value => value != 0.0F))
}

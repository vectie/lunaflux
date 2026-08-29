///|
let probe_input_values : FixedArray[Float] = [
  1.0F,
  -2.0F,
  0.5F,
  4.0F,
  -1.0F,
  3.0F,
  2.0F,
  -0.5F,
  0.25F,
  -4.0F,
  1.5F,
  2.0F,
]

///|
let probe_weight_codes : FixedArray[Int] = [
  127, -3, 0, 8, -127, 7, 2, -1, 5, 0, -9, 11, 1, 2, 3, 4,
]

///|
let probe_scale_values : FixedArray[Float] = [
  0.0078125F, 0.03125F, 0.125F, 0.5F,
]

///|
fn probe_bf16_bits(value : Float) -> Int {
  let bits = value.reinterpret_as_uint()
  ((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16).reinterpret_as_int()
}

///|
fn probe_bf16_float(bits : Int) -> Float {
  Float::reinterpret_from_uint(bits.reinterpret_as_uint() << 16)
}

///|
fn probe_write_i32_le(
  output : FixedArray[Byte],
  index : Int,
  value : Int,
) -> Unit {
  let offset = index * 4
  output[offset] = value.to_byte()
  output[offset + 1] = (value >> 8).to_byte()
  output[offset + 2] = (value >> 16).to_byte()
  output[offset + 3] = (value >> 24).to_byte()
}

///|
fn probe_write_bf16_le(
  output : FixedArray[Byte],
  index : Int,
  value : Float,
) -> Unit {
  let bits = probe_bf16_bits(value)
  let offset = index * 2
  output[offset] = bits.to_byte()
  output[offset + 1] = (bits >> 8).to_byte()
}

///|
fn probe_write_f32_le(
  output : FixedArray[Byte],
  index : Int,
  value : Float,
) -> Unit {
  let bits = value.reinterpret_as_uint()
  let offset = index * 4
  output[offset] = bits.to_byte()
  output[offset + 1] = (bits >> 8).to_byte()
  output[offset + 2] = (bits >> 16).to_byte()
  output[offset + 3] = (bits >> 24).to_byte()
}

///|
fn probe_counts_bytes() -> FixedArray[Byte] {
  let output = FixedArray::make(20, b'\x00')
  for index, value in [1, 1, 2, 3, 3] {
    probe_write_i32_le(output, index, value)
  }
  output
}

///|
fn probe_input_bytes() -> FixedArray[Byte] {
  let output = FixedArray::make(24, b'\x00')
  for index, value in probe_input_values {
    probe_write_bf16_le(output, index, value)
  }
  output
}

///|
fn probe_weight_bytes() -> FixedArray[Byte] {
  let output = FixedArray::make(16, b'\x00')
  for index, value in probe_weight_codes {
    output[index] = value.to_byte()
  }
  output
}

///|
fn probe_scale_bytes() -> FixedArray[Byte] {
  let output = FixedArray::make(16, b'\x00')
  for index, value in probe_scale_values {
    probe_write_f32_le(output, index, value)
  }
  output
}

///|
const TENSOR_CORE_BF16_BYTES : Int64 = 32768L

///|
const TENSOR_CORE_F32_BYTES : Int64 = 65536L

///|
const TENSOR_CORE_ACCUMULATOR_BYTES : Int64 = 1024L

///|
fn tensor_core_bf16_bits(value : Float) -> Int {
  let bits = value.reinterpret_as_uint()
  ((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16).reinterpret_as_int()
}

///|
fn tensor_core_bf16_float(bits : Int) -> Float {
  Float::reinterpret_from_uint(bits.reinterpret_as_uint() << 16)
}

///|
fn tensor_core_write_bf16(
  output : FixedArray[Byte],
  index : Int,
  value : Float,
) -> Unit {
  let bits = tensor_core_bf16_bits(value)
  output[index * 2] = bits.to_byte()
  output[index * 2 + 1] = (bits >> 8).to_byte()
}

///|
fn tensor_core_write_f32(
  output : FixedArray[Byte],
  index : Int,
  value : Float,
) -> Unit {
  let bits = value.reinterpret_as_uint()
  output[index * 4] = bits.to_byte()
  output[index * 4 + 1] = (bits >> 8).to_byte()
  output[index * 4 + 2] = (bits >> 16).to_byte()
  output[index * 4 + 3] = (bits >> 24).to_byte()
}

///|
fn tensor_core_read_f32(input : Bytes, index : Int) -> Float {
  let offset = index * 4
  let bits = input[offset].to_uint() |
    (input[offset + 1].to_uint() << 8) |
    (input[offset + 2].to_uint() << 16) |
    (input[offset + 3].to_uint() << 24)
  Float::reinterpret_from_uint(bits)
}

///|
fn tensor_core_hostile_value(index : Int, cycle : Int, right : Bool) -> Float {
  let phase = if right { 19 } else { 7 }
  match (index + cycle * 13 + phase) % 17 {
    0 => 0.0F
    1 => -0.0F
    2 => 1.5F
    3 => -2.25F
    4 => 0.0078125F
    5 => -0.015625F
    _ =>
      Float::from_int(
        (index * (if right { 11 } else { 5 }) + cycle * 3 + phase) % 29 - 14,
      ) *
      0.0625F
  }
}

///|
fn tensor_core_hostile_bf16(cycle : Int, right : Bool) -> FixedArray[Byte] {
  let output = FixedArray::make(TENSOR_CORE_BF16_BYTES.to_int(), b'\x00')
  for index in 0..<16384 {
    tensor_core_write_bf16(
      output,
      index,
      tensor_core_hostile_value(index, cycle, right),
    )
  }
  output
}

///|
fn tensor_core_accumulator(cycle : Int) -> FixedArray[Byte] {
  let output = FixedArray::make(TENSOR_CORE_ACCUMULATOR_BYTES.to_int(), b'\x00')
  for index in 0..<256 {
    let value = Float::from_int((index * 3 + cycle * 5) % 19 - 9) * 0.03125F
    tensor_core_write_f32(output, index, value)
  }
  output
}

///|
fn tensor_core_cpu_oracle(cycle : Int) -> FixedArray[Float] {
  let output = FixedArray::make(16384, 0.0F)
  for tile in 0..<64 {
    for row in 0..<16 {
      for column in 0..<16 {
        let cell = row * 16 + column
        let index = tile * 256 + cell
        let mut sum = Float::from_int((cell * 3 + cycle * 5) % 19 - 9) *
          0.03125F
        for inner in 0..<16 {
          let left = tensor_core_bf16_float(
            tensor_core_bf16_bits(
              tensor_core_hostile_value(
                tile * 256 + row * 16 + inner,
                cycle,
                false,
              ),
            ),
          )
          let right = tensor_core_bf16_float(
            tensor_core_bf16_bits(
              tensor_core_hostile_value(
                tile * 256 + inner * 16 + column,
                cycle,
                true,
              ),
            ),
          )
          sum = sum + left * right
        }
        output[index] = sum
      }
    }
  }
  output
}

///|
fn validate_tensor_core_output(
  actual : Bytes,
  expected : FixedArray[Float],
  label : String,
) -> Float raise {
  require_tensor_core_probe(
    actual.length() == TENSOR_CORE_F32_BYTES.to_int(),
    label + " output byte count drifted",
  )
  let mut maximum = 0.0F
  for index in 0..<16384 {
    let observed = tensor_core_read_f32(actual, index)
    let reference = expected[index]
    require_tensor_core_probe(observed == observed, label + " produced NaN")
    let error = (observed - reference).abs()
    let tolerance = 0.001F + 0.0001F * reference.abs()
    require_tensor_core_probe(
      error <= tolerance,
      label + " exceeded the bounded F32 tolerance",
    )
    maximum = maximum.max(error)
  }
  maximum
}

///|
fn matrix_bf16_bits(value : Float) -> Int {
  let bits = value.reinterpret_as_uint()
  ((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16).reinterpret_as_int()
}

///|
fn matrix_bf16_float(bits : Int) -> Float {
  Float::reinterpret_from_uint(bits.reinterpret_as_uint() << 16)
}

///|
fn matrix_write_i32(
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
fn matrix_write_bf16(
  output : FixedArray[Byte],
  index : Int,
  value : Float,
) -> Unit {
  let bits = matrix_bf16_bits(value)
  output[index * 2] = bits.to_byte()
  output[index * 2 + 1] = (bits >> 8).to_byte()
}

///|
fn matrix_fill_i32(output : FixedArray[Byte], values : FixedArray[Int]) -> Unit {
  for index, value in values {
    matrix_write_i32(output, index, value)
  }
}

///|
fn matrix_fill_weight(
  output : FixedArray[Byte],
  rows : Int,
  columns : Int,
  scale : Float,
  phase : Int,
) -> Unit {
  for row in 0..<rows {
    for column in 0..<columns {
      let signed = (row * 7 + column * 3 + phase) % 11 - 5
      matrix_write_bf16(
        output,
        row * columns + column,
        Float::from_int(signed) * scale,
      )
    }
  }
}

///|
fn matrix_fill_cache(output : FixedArray[Byte], key~ : Bool) -> Unit {
  for page in 0..<4 {
    for token in 0..<2 {
      for component in 0..<2 {
        let value = if key {
          Float::from_int(page + 1) * 0.125F +
          Float::from_int(token) * 0.0625F +
          Float::from_int(component) * 0.03125F
        } else {
          Float::from_int(page + 1) * 0.5F +
          Float::from_int(token) * 0.125F -
          Float::from_int(component) * 0.0625F
        }
        matrix_write_bf16(output, page * 128 + token * 2 + component, value)
      }
    }
  }
}

///|
fn matrix_poison(output : FixedArray[Byte]) -> Unit {
  for index in 0..<(output.length() / 2) {
    matrix_write_bf16(output, index, 91.0F)
  }
}

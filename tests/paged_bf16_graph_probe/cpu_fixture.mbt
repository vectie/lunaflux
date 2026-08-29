///|
fn bf16_bits(value : Float) -> Int {
  let bits = value.reinterpret_as_uint()
  ((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16).reinterpret_as_int()
}

///|
fn bf16_float(bits : Int) -> Float {
  Float::reinterpret_from_uint(bits.reinterpret_as_uint() << 16)
}

///|
fn ordered_mul(left : Float, right : Float) -> Float {
  left * right
}

///|
fn ordered_add(left : Float, right : Float) -> Float {
  left + right
}

///|
fn write_i32_le(output : FixedArray[Byte], index : Int, value : Int) -> Unit {
  let offset = index * 4
  output[offset] = value.to_byte()
  output[offset + 1] = (value >> 8).to_byte()
  output[offset + 2] = (value >> 16).to_byte()
  output[offset + 3] = (value >> 24).to_byte()
}

///|
fn write_bf16_le(output : FixedArray[Byte], index : Int, value : Float) -> Unit {
  let bits = bf16_bits(value)
  let offset = index * 2
  output[offset] = bits.to_byte()
  output[offset + 1] = (bits >> 8).to_byte()
}

///|
fn fill_weight(
  output : FixedArray[Byte],
  rows : Int,
  columns : Int,
  scale : Float,
  phase : Int,
) -> Unit {
  for row in 0..<rows {
    for column in 0..<columns {
      let signed = (row * 7 + column * 3 + phase) % 11 - 5
      write_bf16_le(
        output,
        row * columns + column,
        Float::from_int(signed) * scale,
      )
    }
  }
}

///|
fn fill_i32_values(output : FixedArray[Byte], values : ArrayView[Int]) -> Unit {
  for index, value in values {
    write_i32_le(output, index, value)
  }
}

///|
fn fill_bf16_values(
  output : FixedArray[Byte],
  values : ArrayView[Float],
) -> Unit {
  for index, value in values {
    write_bf16_le(output, index, value)
  }
}

///|
fn fill_graph_fixture(kind : Int, output : FixedArray[Byte]) -> Bool {
  for index in 0..<output.length() {
    output[index] = b'\x00'
  }
  match kind {
    0 =>
      if output.length() == 20 {
        fill_i32_values(output, [1, 1, 2, 3, 3])
        true
      } else {
        false
      }
    1 =>
      if output.length() == 12 {
        fill_i32_values(output, [2, 5, 7])
        true
      } else {
        false
      }
    2 => {
      if output.length() != 128 {
        return false
      }
      for row in 0..<16 {
        for column in 0..<4 {
          write_bf16_le(
            output,
            row * 4 + column,
            Float::from_int(row - 8) * 0.125F +
            Float::from_int(column) * 0.0625F,
          )
        }
      }
      true
    }
    3 =>
      if output.length() == 8 {
        fill_bf16_values(output, [1.0F, 0.75F, 1.25F, 0.5F])
        true
      } else {
        false
      }
    4 =>
      if output.length() == 32 {
        fill_weight(output, 4, 4, 0.125F, 1)
        true
      } else {
        false
      }
    5 =>
      if output.length() == 16 {
        fill_weight(output, 2, 4, 0.25F, 2)
        true
      } else {
        false
      }
    6 =>
      if output.length() == 16 {
        fill_weight(output, 2, 4, 0.0625F, 3)
        true
      } else {
        false
      }
    7 =>
      if output.length() == 12 {
        fill_i32_values(output, [0, 1, 2])
        true
      } else {
        false
      }
    8 =>
      if output.length() == 12 {
        fill_i32_values(output, [0, 2, 3])
        true
      } else {
        false
      }
    9 =>
      if output.length() == 8 {
        fill_i32_values(output, [2, 3])
        true
      } else {
        false
      }
    10 =>
      if output.length() == 12 {
        fill_i32_values(output, [0, 1, 3])
        true
      } else {
        false
      }
    11 =>
      if output.length() == 12 {
        fill_i32_values(output, [0, 1, 2])
        true
      } else {
        false
      }
    12 | 13 => {
      if output.length() != 1024 {
        return false
      }
      let values : ArrayView[Float] = if kind == 12 {
        [0.5F, -0.25F, -0.5F, 0.25F]
      } else {
        [1.5F, 0.75F, 2.0F, -1.0F]
      }
      for index, value in values {
        write_bf16_le(output, 128 + index, value)
      }
      true
    }
    14 =>
      if output.length() == 32 {
        fill_weight(output, 4, 4, 0.125F, 7)
        true
      } else {
        false
      }
    15 =>
      if output.length() == 8 {
        fill_bf16_values(output, [1.125F, 0.875F, 0.75F, 1.25F])
        true
      } else {
        false
      }
    16 =>
      if output.length() == 64 {
        fill_weight(output, 8, 4, 0.125F, 4)
        true
      } else {
        false
      }
    17 =>
      if output.length() == 64 {
        fill_weight(output, 8, 4, 0.0625F, 5)
        true
      } else {
        false
      }
    18 =>
      if output.length() == 64 {
        fill_weight(output, 4, 8, 0.125F, 6)
        true
      } else {
        false
      }
    19 =>
      if output.length() == 8 {
        fill_bf16_values(output, [0.8125F, 1.1875F, 1.0F, 0.625F])
        true
      } else {
        false
      }
    20 =>
      if output.length() == 128 {
        fill_weight(output, 16, 4, 0.125F, 8)
        true
      } else {
        false
      }
    21 => {
      if output.length() % 2 != 0 {
        return false
      }
      for index in 0..<(output.length() / 2) {
        write_bf16_le(output, index, 91.0F)
      }
      true
    }
    _ => false
  }
}

///|
fn fixture_bf16(kind : Int, elements : Int) -> FixedArray[Int] {
  let bytes = FixedArray::make(elements * 2, b'\x00')
  try! assert_true(fill_graph_fixture(kind, bytes))
  FixedArray::makei(elements, index => {
    bytes[index * 2].to_int() | (bytes[index * 2 + 1].to_int() << 8)
  })
}

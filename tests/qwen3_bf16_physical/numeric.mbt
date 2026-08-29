///|
fn qwen3_bf16_bits(value : Float) -> Int {
  let bits = value.reinterpret_as_uint()
  ((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16).reinterpret_as_int()
}

///|
fn qwen3_bf16_float(bits : Int) -> Float {
  Float::reinterpret_from_uint(bits.reinterpret_as_uint() << 16)
}

///|
fn qwen3_write_bf16(
  output : FixedArray[Byte],
  index : Int,
  value : Float,
) -> Unit {
  let bits = qwen3_bf16_bits(value)
  output[index * 2] = bits.to_byte()
  output[index * 2 + 1] = (bits >> 8).to_byte()
}

///|
fn qwen3_read_bf16(input : Bytes, index : Int) -> Float {
  let bits = input[index * 2].to_int() | (input[index * 2 + 1].to_int() << 8)
  qwen3_bf16_float(bits)
}

///|
fn qwen3_i32_counts() -> Bytes {
  let output = FixedArray::make(20, b'\x00')
  output[12] = b'\x01'
  Bytes::from_array(output[:])
}

///|
fn qwen3_input_fixture(width : Int) -> Bytes {
  let output = FixedArray::make(width * 2, b'\x00')
  for index in 0..<width {
    qwen3_write_bf16(output, index, Float::from_int(index % 13 - 6) * 0.0625F)
  }
  Bytes::from_array(output[:])
}

///|
fn qwen3_sparse_weight(rows : Int, columns : Int, phase : Int) -> Bytes {
  let output = FixedArray::make(rows * columns * 2, b'\x00')
  for row in 0..<rows {
    let scale = Float::from_int((row + phase) % 7 - 3) * 0.125F
    qwen3_write_bf16(output, row * columns + row % columns, scale)
  }
  Bytes::from_array(output[:])
}

///|
fn qwen3_norm_weight(width : Int, phase : Int) -> Bytes {
  let output = FixedArray::make(width * 2, b'\x00')
  for index in 0..<width {
    qwen3_write_bf16(
      output,
      index,
      0.75F + Float::from_int((index + phase) % 5) * 0.0625F,
    )
  }
  Bytes::from_array(output[:])
}

///|
fn qwen3_expected_qkv(
  input : Bytes,
  query_weight : Bytes,
  key_weight : Bytes,
  value_weight : Bytes,
  hidden : Int,
  query_width : Int,
  kv_width : Int,
) -> FixedArray[Int] {
  let output = FixedArray::make(query_width + 2 * kv_width, 0)
  for column in 0..<output.length() {
    let weight = if column < query_width {
      query_weight
    } else if column < query_width + kv_width {
      key_weight
    } else {
      value_weight
    }
    let local_column = if column < query_width {
      column
    } else if column < query_width + kv_width {
      column - query_width
    } else {
      column - query_width - kv_width
    }
    let inner = local_column % hidden
    output[column] = qwen3_bf16_bits(
      qwen3_read_bf16(input, inner) *
      qwen3_read_bf16(weight, local_column * hidden + inner),
    )
  }
  output
}

///|
fn qwen3_compare_qkv(
  actual : Bytes,
  expected : FixedArray[Int],
) -> Unit raise Qwen3PhysicalError {
  qwen3_require(actual.length() == expected.length() * 2, "QKV size differs")
  for index in 0..<expected.length() {
    let actual_bits = qwen3_bf16_bits(qwen3_read_bf16(actual, index))
    if actual_bits != expected[index] {
      raise CandidateMismatch("independent-width QKV numerical result differs")
    }
  }
}

///|
fn qwen3_compare_qk_norm(
  input : Bytes,
  actual : Bytes,
  q_norm : Bytes,
  k_norm : Bytes,
  query_heads : Int,
  kv_heads : Int,
  head_dimension : Int,
  epsilon : Float,
) -> Float raise Qwen3PhysicalError {
  let packed_heads = query_heads + 2 * kv_heads
  qwen3_require(
    input.length() == packed_heads * head_dimension * 2 &&
    actual.length() == input.length(),
    "Q/K RMS normalization size differs",
  )
  let mut maximum_error = 0.0F
  for head in 0..<packed_heads {
    let base = head * head_dimension
    if head >= query_heads + kv_heads {
      for lane in 0..<head_dimension {
        qwen3_require(
          qwen3_bf16_bits(qwen3_read_bf16(actual, base + lane)) ==
          qwen3_bf16_bits(qwen3_read_bf16(input, base + lane)),
          "Q/K RMS value-head passthrough differs",
        )
      }
    } else {
      let mut squares = 0.0F
      for lane in 0..<head_dimension {
        let value = qwen3_read_bf16(input, base + lane)
        squares += value * value
      }
      let inverse = 1.0F /
        (squares / Float::from_int(head_dimension) + epsilon).sqrt()
      let weight = if head < query_heads { q_norm } else { k_norm }
      for lane in 0..<head_dimension {
        let expected = qwen3_bf16_float(
          qwen3_bf16_bits(
            qwen3_read_bf16(input, base + lane) *
            inverse *
            qwen3_read_bf16(weight, lane),
          ),
        )
        let observed = qwen3_read_bf16(actual, base + lane)
        let error = (observed - expected).abs()
        maximum_error = maximum_error.max(error)
        qwen3_require(
          error <= 0.02F + expected.abs() * 0.005F,
          "Q/K RMS normalization numerical result differs",
        )
      }
    }
  }
  maximum_error
}

///|
test "Qwen independent-width sparse QKV referee separates K and V" {
  let input = qwen3_input_fixture(4)
  let q = qwen3_sparse_weight(8, 4, 1)
  let k = qwen3_sparse_weight(4, 4, 2)
  let v = qwen3_sparse_weight(4, 4, 3)
  let expected = qwen3_expected_qkv(input, q, k, v, 4, 8, 4)
  assert_eq(expected.length(), 16)
  assert_not_eq(expected[8], expected[12])
}

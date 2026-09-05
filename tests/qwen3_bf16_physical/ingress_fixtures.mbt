///|
fn ingress_i32(values : ArrayView[Int]) -> Bytes {
  let bytes = FixedArray::make(values.length() * 4, b'\x00')
  for index, value in values {
    for shift in 0..<4 {
      bytes[index * 4 + shift] = (value >> (shift * 8)).to_byte()
    }
  }
  Bytes::from_array(bytes[:])
}

///|
// Dense dyadic fixtures make every product and dot sum exactly representable
// in FP32. This isolates topology/indexing regressions from allowed rounding
// changes. It is deliberately not a trained-model accuracy qualification.
fn ingress_dense(rows : Int, columns : Int, phase : Int) -> Bytes {
  let bytes = FixedArray::make(rows * columns * 2, b'\x00')
  for row in 0..<rows {
    for column in 0..<columns {
      qwen3_write_bf16(
        bytes,
        row * columns + column,
        Float::from_int((row * 3 + column * 7 + phase) % 17 - 8) / 64.0F,
      )
    }
  }
  Bytes::from_array(bytes[:])
}

///|
fn ingress_reference_dot(
  input : Bytes,
  weight : Bytes,
  row : Int,
  column : Int,
) -> Float {
  for inner in 0..<1024; sum = 0.0F {
    continue sum +
      qwen3_read_bf16(input, row * 1024 + inner) *
      qwen3_read_bf16(weight, column * 1024 + inner)
  } nobreak {
    qwen3_bf16_float(qwen3_bf16_bits(sum))
  }
}

///|
fn ingress_validate(
  actual : ArrayView[Bytes],
  reference : ArrayView[Bytes],
  input : Bytes,
  value_weight : Bytes,
  tokens : Int,
) -> Unit raise {
  assert_eq(actual.length(), 3)
  // Includes untouched output rows and KV pages initialized to a sentinel.
  for index in 0..<3 {
    assert_eq(actual[index], reference[index])
  }
  for token in 0..<tokens {
    // Independent dense dot oracle samples all eight V heads, not just head 0.
    for
      column in [
        0, 127, 128, 255, 256, 383, 384, 511, 512, 639, 640, 767, 768, 895, 896,
        1023,
      ] {
      assert_eq(
        qwen3_read_bf16(actual[0], token * 4096 + 3072 + column),
        ingress_reference_dot(input, value_weight, token, column),
      )
    }
    for column in 0..<1024 {
      // Page table reverses physical pages; one prefix token remains untouched.
      let position = token + 1
      let cache_index = (32 - position / 8) * 8192 +
        position % 8 * 1024 +
        column
      assert_eq(
        qwen3_read_bf16(actual[1], cache_index),
        qwen3_read_bf16(actual[0], token * 4096 + 2048 + column),
      )
      assert_eq(
        qwen3_read_bf16(actual[2], cache_index),
        qwen3_read_bf16(actual[0], token * 4096 + 3072 + column),
      )
    }
  }
  assert_eq(qwen3_read_bf16(actual[1], 32 * 8192), 0.25F)
  assert_eq(qwen3_read_bf16(actual[2], 32 * 8192), 0.25F)
}

///|
test "ingress dense fixture covers cancellation and independent value heads" {
  let input = ingress_dense(1, 1024, 2)
  let weight = ingress_dense(1024, 1024, 5)
  assert_not_eq(
    ingress_reference_dot(input, weight, 0, 0),
    ingress_reference_dot(input, weight, 0, 127),
  )
  assert_eq(ingress_i32([256, 1]), b"\x00\x01\x00\x00\x01\x00\x00\x00")
}

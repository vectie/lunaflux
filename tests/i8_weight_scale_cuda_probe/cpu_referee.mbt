///|
fn probe_ordered_mul(left : Float, right : Float) -> Float {
  left * right
}

///|
fn probe_ordered_add(left : Float, right : Float) -> Float {
  left + right
}

///|
fn expected_i8_projection_bf16() -> FixedArray[Int] {
  FixedArray::makei(12, index => {
    let row = index / 4
    let output_channel = index % 4
    let mut accumulator = 0.0F
    for inner in 0..<4 {
      let reconstructed = probe_ordered_mul(
        Float::from_int(probe_weight_codes[output_channel * 4 + inner]),
        probe_scale_values[output_channel],
      )
      accumulator = probe_ordered_add(
        accumulator,
        probe_ordered_mul(probe_input_values[row * 4 + inner], reconstructed),
      )
    }
    probe_bf16_bits(accumulator)
  })
}

///|
fn validate_i8_projection_output(
  actual : Bytes,
) -> Double raise I8PhysicalProbeFailure {
  let expected = expected_i8_projection_bf16()
  require_probe(actual.length() == 24, "I8 output byte count changed")
  let mut maximum_error = 0.0
  for index, expected_bits in expected {
    let actual_bits = actual[index * 2].to_int() |
      (actual[index * 2 + 1].to_int() << 8)
    let actual_value = probe_bf16_float(actual_bits)
    let expected_value = probe_bf16_float(expected_bits)
    let error = (actual_value - expected_value).abs()
    require_probe(
      !error.is_nan() && !error.is_inf() && error <= 0.0F,
      "independent I8 numeric referee rejected output \{index}",
    )
    maximum_error = maximum_error.max(error.to_double())
  }
  maximum_error
}

///|
test "independent referee covers reserved-edge-adjacent I8 codes and per-row scales" {
  assert_true(probe_weight_codes.contains(-127))
  assert_true(probe_weight_codes.contains(127))
  assert_false(probe_weight_codes.contains(-128))
  let expected = expected_i8_projection_bf16()
  assert_eq(expected.length(), 12)
  assert_true(expected.any(value => value != 0))
}

///|
test "E4M3FN and UE8M0 scalar interpretations cover exact boundaries" {
  assert_eq(decode_e4m3_finite(0x00), 0.0F)
  assert_eq(decode_e4m3_finite(0x01), 0.001953125F)
  assert_eq(decode_e4m3_finite(0x38), 1.0F)
  assert_eq(decode_e4m3_finite(0x7e), 448.0F)
  assert_eq(decode_e4m3_finite(0xfe), -448.0F)
  assert_eq(decode_ue8m0(127), 1.0F)
  assert_eq(decode_ue8m0(128), 2.0F)
  try decode_ue8m0(255) catch {
    InvalidE8M0Code(255) => ()
    _ => fail("wrong UE8M0 failure")
  } noraise {
    _ => fail("UE8M0 NaN admitted")
  }
}

///|
test "activation scaling rounds upward and dot is block-local" {
  let values = Array::make(128, 0.0F)
  values[0] = 448.0F
  assert_eq(activation_scale_ue8m0_code(values), 127)
  let weights = Array::make(128, 0x38)
  let result = block128_dot_bf16_boundary(values, weights, [127])
  assert_eq(result, 448.0F)
  try block128_dot_bf16_boundary(values, weights, [127, 127]) catch {
    InvalidShape => ()
    _ => fail("wrong shape failure")
  } noraise {
    _ => fail("wrong scale count admitted")
  }
}

///|
test "512-wide projection dot preserves four ordered block boundaries" {
  let values = Array::make(512, 0.0F)
  let weights = Array::make(512, 0x38)
  for block in 0..<4 {
    values[block * 128] = Float::from_int(block + 1)
  }
  assert_eq(
    block128_dot_bf16_boundary(values, weights, [127, 127, 127, 127]),
    10.0F,
  )
}

///|
test "rotary startup chunks preserve F32 little endian bits and slice boundaries" {
  let values : Array[Float] = [0.0F, -0.0F, 1.0F, -1.0F, 0.5F]
  let whole = rotary_chunk(values)
  assert_eq(whole.length(), 20)
  assert_eq(whole[7], 128)
  assert_eq(whole[10], 128)
  assert_eq(whole[11], 63)
  assert_eq(whole[15], 191)
  assert_eq(rotary_chunk(values[:2]) + rotary_chunk(values[2:]), whole)
  assert_eq(rotary_chunk([]).length(), 0)
}

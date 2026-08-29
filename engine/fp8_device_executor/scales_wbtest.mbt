///|
fn write_fp8_test_u32(
  destination : FixedArray[Byte],
  offset : Int,
  bits : UInt,
) -> Unit {
  destination[offset] = (bits & 0xffU).to_byte()
  destination[offset + 1] = ((bits >> 8) & 0xffU).to_byte()
  destination[offset + 2] = ((bits >> 16) & 0xffU).to_byte()
  destination[offset + 3] = ((bits >> 24) & 0xffU).to_byte()
}

///|
test "scale publication admits every independent finite-positive stage" {
  let scratch = FixedArray::make(8, b'\x00')
  write_fp8_test_u32(scratch, 0, 0x3f800000U)
  write_fp8_test_u32(scratch, 4, 0x40000000U)
  assert_true(validate_fp8_scale_scratch(scratch, 4L))
  assert_true(validate_fp8_scale_scratch(scratch, 8L))
}

///|
test "canonical sentinel and every nonfinite or nonpositive scale reject" {
  let hostile : Array[UInt] = [
    0x7fc00000U, 0x7f800000U, 0xff800000U, 0x00000000U, 0x80000000U, 0xbf800000U,
  ]
  for bits in hostile {
    let scratch = FixedArray::make(8, b'\x00')
    write_fp8_test_u32(scratch, 0, bits)
    write_fp8_test_u32(scratch, 4, 0x3f800000U)
    assert_false(validate_fp8_scale_scratch(scratch, 4L))
    assert_false(validate_fp8_scale_scratch(scratch, 8L))
  }
}

///|
test "gated evidence rejects either stale independent stage" {
  let scratch = FixedArray::make(8, b'\x00')
  write_fp8_test_u32(scratch, 0, 0x3f000000U)
  write_fp8_test_u32(scratch, 4, 0x7fc00000U)
  assert_true(validate_fp8_scale_scratch(scratch, 4L))
  assert_false(validate_fp8_scale_scratch(scratch, 8L))
  write_fp8_test_u32(scratch, 0, 0x7fc00000U)
  write_fp8_test_u32(scratch, 4, 0x3f000000U)
  assert_false(validate_fp8_scale_scratch(scratch, 8L))
}

///|
test "scale publication rejects malformed workspace widths" {
  let scratch = FixedArray::make(8, b'\x00')
  write_fp8_test_u32(scratch, 0, 0x3f800000U)
  write_fp8_test_u32(scratch, 4, 0x3f800000U)
  assert_false(validate_fp8_scale_scratch(scratch, 0L))
  assert_false(validate_fp8_scale_scratch(scratch, 5L))
  assert_false(validate_fp8_scale_scratch(scratch, 12L))
}

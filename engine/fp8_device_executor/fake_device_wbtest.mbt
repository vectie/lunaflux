///|
test "fake device launch failure never permits output publication" {
  let scratch = FixedArray::make(8, b'\x00')
  write_fp8_test_u32(scratch, 0, 0x3f800000U)
  assert_false(admit_fp8_execution_evidence(false, true, scratch, 4L))
}

///|
test "fake device readback failure never permits output publication" {
  let scratch = FixedArray::make(8, b'\x00')
  write_fp8_test_u32(scratch, 0, 0x3f800000U)
  assert_false(admit_fp8_execution_evidence(true, false, scratch, 4L))
}

///|
test "fake device stale sentinel never permits output publication" {
  let scratch = FixedArray::make(8, b'\x00')
  write_fp8_test_u32(scratch, 0, 0x7fc00000U)
  assert_false(admit_fp8_execution_evidence(true, true, scratch, 4L))
}

///|
test "fake gated device must publish both independent scale cells" {
  let scratch = FixedArray::make(8, b'\x00')
  write_fp8_test_u32(scratch, 0, 0x3f000000U)
  write_fp8_test_u32(scratch, 4, 0x7fc00000U)
  assert_false(admit_fp8_execution_evidence(true, true, scratch, 8L))
  write_fp8_test_u32(scratch, 4, 0x40000000U)
  assert_true(admit_fp8_execution_evidence(true, true, scratch, 8L))
}

///|
fn write_fp8_scale_test_bits(
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
test "reusable FP8 scale arena arms every independent cell with qNaN" {
  let evidence = FixedArray::make(20, b'\x5a')
  fill_fp8_scale_sentinel_v3(evidence)
  for cell in 0..<5 {
    assert_eq(fp8_scale_read_u32(evidence, cell * 4), 0x7fc00000U)
    assert_false(fp8_scale_cell_valid(fp8_scale_read_u32(evidence, cell * 4)))
  }
}

///|
test "an early FP8 sentinel cannot be hidden by later finite publications" {
  let evidence = FixedArray::make(16, b'\x00')
  fill_fp8_scale_sentinel_v3(evidence)
  // Simulate a failed first simple projection followed by one successful gated
  // operation and one successful simple projection in their disjoint regions.
  write_fp8_scale_test_bits(evidence, 4, 0x3f800000U)
  write_fp8_scale_test_bits(evidence, 8, 0x40000000U)
  write_fp8_scale_test_bits(evidence, 12, 0x3f000000U)
  assert_false(fp8_scale_cell_valid(fp8_scale_read_u32(evidence, 0)))
  for cell in 1..<4 {
    assert_true(fp8_scale_cell_valid(fp8_scale_read_u32(evidence, cell * 4)))
  }
  let executor = dummy_paged_executor(GraphExecuting)
  poison_fp8_graph_lifecycle(executor)
  assert_eq(executor.lifecycle(), GraphPoisoned)
  try executor.stage(executor_test_submitted_plan()) catch {
    InvalidExecutorLifecycle(expected=GraphReady, actual=GraphPoisoned) => ()
    _ => fail("unexpected poisoned FP8 executor reuse rejection")
  } noraise {
    _ => fail("early sentinel failure left the FP8 executor reusable")
  }
}

///|
test "FP8 scale evidence rejects every nonpositive or nonfinite encoding" {
  for
    bits in [
      0x00000000U, 0x80000000U, 0xbf800000U, 0x7f800000U, 0xff800000U, 0x7fc00000U,
    ] {
    assert_false(fp8_scale_cell_valid(bits))
  }
  assert_true(fp8_scale_cell_valid(0x00000001U))
  assert_true(fp8_scale_cell_valid(0x3f800000U))
}

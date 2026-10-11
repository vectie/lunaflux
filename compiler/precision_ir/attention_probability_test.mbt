///|
test "online BF16 probabilities are explicit validated numerical semantics" {
  let make = fn(law) raise {
    @precision_ir.WindowSharedKvPrecision::new(
      query_heads=2,
      head_width=8,
      window_tokens=128,
      maximum_rows=144,
      maximum_positions=512,
      maximum_state_bytes=10000L,
      probability=law,
    )
  }
  assert_eq(make(GlobalF32).probability_law(), GlobalF32)
  for block in [1, 64, 4096] {
    assert_eq(
      make(OnlineBf16(block_tokens=block)).probability_law(),
      OnlineBf16(block_tokens=block),
    )
  }
  for block in [0, 3, 8192] {
    try make(OnlineBf16(block_tokens=block)) catch {
      @precision_ir.PrecisionPlanError::IllegalPlacement => ()
      error => raise error
    } noraise {
      _ => fail("invalid numerical partition accepted")
    }
  }
}

///|
test "shared window KV storage is bounded by window not context" {
  let plan = @precision_ir.WindowSharedKvPrecision::new(
    query_heads=64,
    head_width=512,
    window_tokens=128,
    maximum_rows=16,
    maximum_positions=1048576,
    maximum_state_bytes=1179668L,
  )
  assert_eq(plan.state_bytes(), 1179668L)
  assert_eq(plan.buffer_bytes()[0], 131072L)
  assert_eq(plan.buffer_bytes()[3], 1048576L)
  assert_eq(plan.effect(), @precision_ir.NumericEffect::CacheCommit)
  try
    @precision_ir.WindowSharedKvPrecision::new(
      query_heads=64,
      head_width=512,
      window_tokens=128,
      maximum_rows=16,
      maximum_positions=1048576,
      maximum_state_bytes=1179667L,
    )
  catch {
    @precision_ir.PrecisionPlanError::MemoryBudgetExceeded => ()
    error => raise error
  } noraise {
    _ => fail("accepted undersized state budget")
  }
}

///|
test "convolution precision separates frame capacity from request history" {
  let plan = @precision_ir.CausalConvolutionPrecision::new(
    channel_count=8192,
    kernel_size=4,
    maximum_rows=32,
    maximum_sequences=16,
    cache_slots=32,
    maximum_state_bytes=1572864L,
  )
  assert_eq(plan.geometry(), (8192, 4))
  assert_eq(plan.capacity(), (32, 16, 32))
  assert_eq(plan.frame_bytes(), 524288L)
  assert_eq(plan.state_bytes(), 1572864L)
  assert_eq(plan.effect(), @precision_ir.NumericEffect::CacheCommit)
  try
    @precision_ir.CausalConvolutionPrecision::new(
      channel_count=8192,
      kernel_size=4,
      maximum_rows=32,
      maximum_sequences=16,
      cache_slots=32,
      maximum_state_bytes=1572863L,
    )
  catch {
    @precision_ir.PrecisionPlanError::MemoryBudgetExceeded => ()
    e => raise e
  } noraise {
    _ => fail("history exceeded budget")
  }
}

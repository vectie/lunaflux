///|
test "recurrent request slots have an explicit bounded persistent effect" {
  let plan = @precision_ir.RecurrentDeltaPrecision::new(
    head_count=64,
    key_dimension=128,
    value_dimension=128,
    maximum_rows=32,
    maximum_sequences=16,
    cache_slots=32,
    normalization_epsilon=0.000001,
    log_decay_lower_bound=-5.0,
    maximum_state_bytes=134217728L,
  )
  assert_eq(plan.geometry(), (64, 128, 128))
  assert_eq(plan.capacity(), (32, 16, 32))
  assert_eq(plan.state_bytes(), 134217728L)
  assert_eq(plan.effect(), @precision_ir.NumericEffect::CacheCommit)
  try
    @precision_ir.RecurrentDeltaPrecision::new(
      head_count=64,
      key_dimension=128,
      value_dimension=128,
      maximum_rows=32,
      maximum_sequences=16,
      cache_slots=32,
      normalization_epsilon=0.000001,
      log_decay_lower_bound=-5.0,
      maximum_state_bytes=134217727L,
    )
  catch {
    @precision_ir.PrecisionPlanError::MemoryBudgetExceeded => ()
    e => raise e
  } noraise {
    _ => fail("persistent state budget ignored")
  }
}

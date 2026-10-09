///|
test "retained sparse history accounts frame and cache storage independently" {
  let attention = @precision_ir.IndexedSparseAttentionPrecision::new(
    index_head_count=1,
    index_head_dimension=4,
    query_head_count=2,
    key_value_head_count=1,
    query_key_dimension=4,
    value_dimension=4,
    pool_size=4,
    top_k=8,
    append_incomplete_tail=true,
    maximum_queries=8,
    maximum_history=32,
    maximum_workspace_bytes=100000L,
  )
  let plan = @precision_ir.IndexedSparseHistoryPrecision::new(
    attention,
    maximum_state_bytes=1216L,
  )
  assert_eq(plan.attention().capacity(), (8, 32))
  assert_eq(plan.buffer_bytes(), [256L, 256L, 256L, 256L, 128L, 20L, 12L, 32L])
  assert_eq(plan.state_bytes(), 1216L)
  try
    @precision_ir.IndexedSparseHistoryPrecision::new(
      attention,
      maximum_state_bytes=1215L,
    )
  catch {
    @precision_ir.PrecisionPlanError::MemoryBudgetExceeded => ()
    e => raise e
  } noraise {
    _ => fail("request state ceiling ignored")
  }
}

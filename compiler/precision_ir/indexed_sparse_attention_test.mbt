///|
test "decode query capacity is independent of long projected history" {
  let plan = @precision_ir.IndexedSparseAttentionPrecision::new(
    index_head_count=32,
    index_head_dimension=128,
    query_head_count=64,
    key_value_head_count=64,
    query_key_dimension=256,
    value_dimension=256,
    pool_size=4,
    top_k=2048,
    append_incomplete_tail=true,
    maximum_queries=1,
    maximum_history=32768,
    maximum_workspace_bytes=4000000L,
  )
  assert_eq(plan.capacity(), (1, 32768))
  assert_eq(plan.selected_width(), 2051)
  assert_eq(plan.frame_bytes(), [2097152L, 131072L, 32768L, 4L, 8204L])
  assert_eq(plan.workspace_bytes(), 2269200L)
  try
    @precision_ir.IndexedSparseAttentionPrecision::new(
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
      maximum_workspace_bytes=1L,
    )
  catch {
    @precision_ir.PrecisionPlanError::MemoryBudgetExceeded => ()
    e => raise e
  } noraise {
    _ => fail("workspace ceiling ignored")
  }
}

///|
test "rope-free sparse projections account for all intermediates and affine weights" {
  let attention = @precision_ir.IndexedSparseAttentionPrecision::new(
    index_head_count=2,
    index_head_dimension=4,
    query_head_count=2,
    key_value_head_count=2,
    query_key_dimension=4,
    value_dimension=4,
    pool_size=4,
    top_k=8,
    append_incomplete_tail=true,
    maximum_queries=8,
    maximum_history=32,
    maximum_workspace_bytes=100000L,
  )
  let plan = @precision_ir.IndexedSparseProjectionPrecision::new(
    attention,
    hidden_width=8,
    query_rank=4,
    key_value_rank=2,
    rms_epsilon=1.0e-5,
    index_layer_norm_epsilon=1.0e-5,
    maximum_workspace_bytes=1184L,
  )
  assert_eq(plan.workspace_bytes(), 1184L)
  assert_eq(plan.frame_bytes().length(), 13)
  assert_eq(plan.weight_bytes().length(), 12)
  assert_eq(plan.weight_bytes()[8], 8L)
  assert_eq(plan.weight_bytes()[9], 8L)
  try
    @precision_ir.IndexedSparseProjectionPrecision::new(
      attention,
      hidden_width=8,
      query_rank=4,
      key_value_rank=2,
      rms_epsilon=1.0e-5,
      index_layer_norm_epsilon=1.0e-5,
      maximum_workspace_bytes=1183L,
    )
  catch {
    @precision_ir.PrecisionPlanError::MemoryBudgetExceeded => ()
    _ => fail("wrong error")
  } noraise {
    _ => fail("budget ignored")
  }
}

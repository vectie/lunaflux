///|
test "latent cache keeps score normalization and has one storage owner" {
  let base = @precision_ir.IndexedSparseAttentionPrecision::new(
    index_head_count=16,
    index_head_dimension=32,
    query_head_count=4,
    key_value_head_count=2,
    query_key_dimension=32,
    value_dimension=32,
    pool_size=4,
    top_k=32,
    append_incomplete_tail=true,
    maximum_queries=17,
    maximum_history=64,
    maximum_workspace_bytes=100000L,
    attention_accumulation=TiledFloat32,
  )
  let projection = @precision_ir.IndexedSparseProjectionPrecision::new(
    base,
    hidden_width=64,
    query_rank=32,
    key_value_rank=64,
    rms_epsilon=1.0e-5,
    index_layer_norm_epsilon=1.0e-5,
    maximum_workspace_bytes=1000000L,
    cache_law=LatentBFloat16,
  )
  let execution = projection.execution_attention()
  assert_eq(execution.attention_geometry(), (4, 1, 64, 64))
  assert_eq(execution.score_dimension(), 32)
  assert_eq(base.attention_geometry(), (4, 2, 32, 32))
  assert_true(execution.shares_key_value())
  assert_eq(projection.execution_order(), [0, 1, 2, 3, 4, 12, 7, 8, 9, 10, 11])
  assert_eq(projection.frame_owner(7), 4)
  assert_eq(projection.frame_owner(13), 13)
  let history = @precision_ir.IndexedSparseHistoryPrecision::new(
    execution,
    maximum_state_bytes=1000000L,
  )
  assert_eq(history.buffer_owner(3), 2)
  assert_eq(
    history.state_bytes(),
    history.buffer_bytes().fold(init=0L, (sum, size) => sum + size) -
    history.buffer_bytes()[3],
  )
  try
    @precision_ir.IndexedSparseHistoryPrecision::new(
      execution,
      maximum_state_bytes=history.state_bytes() - 1L,
    )
  catch {
    @precision_ir.PrecisionPlanError::MemoryBudgetExceeded => ()
    _ => fail("incorrect budget error")
  } noraise {
    _ => fail("budget ignored")
  }
}

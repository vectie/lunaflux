///|
test "request history publishes after parallel bit-exact append" {
  let attention = @precision.IndexedSparseAttentionPrecision::new(
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
  let source = @indexed_sparse_attention_execution_cuda_source.indexed_sparse_history_source(
    @precision.IndexedSparseHistoryPrecision::new(
      attention,
      maximum_state_bytes=1216L,
    ),
    symbol_prefix="retained",
  )
  assert_eq(source.launch(0), (1, 1))
  assert_eq(source.launch(1), (1, 256))
  assert_eq(source.launch(2), (1, 1))
  assert_eq(source.symbol(0), "retained_begin")
  assert_eq(source.symbol(1), "retained_copy")
  assert_eq(source.symbol(2), "retained_publish")
  let text = @utf8.decode(source.source_bytes())
  assert_true(text.contains("positions[row]=base+row"))
  assert_true(text.contains("history_values[base*vw+i]=values[i]"))
  assert_true(text.contains("history[3]=base+live"))
  assert_true(text.contains("live>32-base"))
  assert_true(!text.contains("atomic"))
}

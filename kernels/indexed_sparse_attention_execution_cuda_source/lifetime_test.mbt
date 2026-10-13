///|
test "index retention aliases only the ordered attention scratch lifetime" {
  let plan = @precision.IndexedSparseAttentionPrecision::new(
    index_head_count=1,
    index_head_dimension=4,
    query_head_count=1,
    key_value_head_count=1,
    query_key_dimension=4,
    value_dimension=4,
    pool_size=1,
    top_k=1,
    append_incomplete_tail=false,
    maximum_queries=16,
    maximum_history=128,
    maximum_workspace_bytes=100000L,
  )
  let source = @indexed_sparse_attention_execution_cuda_source.indexed_sparse_attention_source(
    plan,
    symbol_prefix="lifetime",
  )
  assert_eq(source.scratch_bytes(), 16L * 128L * 8L)
  assert_eq(source.attention_scratch_bytes(), 16L * 8L)
  assert_eq(source.launch(1), (16, 128))
}

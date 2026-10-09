///|
test "projection chain preserves low-rank RMSNorm and affine index LayerNorm" {
  let attention = @precision.IndexedSparseAttentionPrecision::new(
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
  let source = @indexed_sparse_projection_cuda_source.indexed_sparse_projection_source(
    @precision.IndexedSparseProjectionPrecision::new(
      attention,
      hidden_width=8,
      query_rank=4,
      key_value_rank=2,
      rms_epsilon=1.0e-5,
      index_layer_norm_epsilon=1.0e-5,
      maximum_workspace_bytes=1312L,
    ),
    symbol_prefix="sparse_projection",
  )
  assert_eq(source.symbol(9), "sparse_projection_index_norm")
  assert_eq(source.launch(9), (8, 1))
  assert_eq(source.launch(2), (1, 256))
  assert_eq(source.launch(6), (1, 256))
  assert_eq(source.plan().workspace_bytes(), 1184L)
  if @env.get_env_var("LUNA_EXPORT_SPARSE_PROJECTION") is Some("1") {
    println("LUNA_SOURCE_BEGIN projection")
    println(@utf8.decode(source.source_bytes()))
    println(
      @utf8.decode(
        @sparse.indexed_sparse_history_source(
          @precision.IndexedSparseHistoryPrecision::new(
            attention,
            maximum_state_bytes=10000L,
          ),
          symbol_prefix="projected_history",
        ).source_bytes(),
      ),
    )
    println(
      @utf8.decode(
        @sparse.indexed_sparse_attention_source(
          attention,
          symbol_prefix="projected_attention",
        ).source_bytes(),
      ),
    )
    println("LUNA_SOURCE_END")
  }
}

///|
test "selection producer bounds the written prefix without changing storage" {
  for
    (history, top_k, tail, expected) in [
      (64, 2048, true, 67),
      (63, 2048, true, 63),
      (4, 2048, true, 7),
      (4, 2048, false, 4),
      (4096, 2048, true, 2051),
      (4096, 2048, false, 2048),
    ] {
    let semantic = @precision.IndexedSparseAttentionPrecision::new(
      index_head_count=1,
      index_head_dimension=4,
      query_head_count=2,
      key_value_head_count=1,
      query_key_dimension=4,
      value_dimension=4,
      pool_size=4,
      top_k~,
      append_incomplete_tail=tail,
      maximum_queries=1,
      maximum_history=history,
      maximum_workspace_bytes=10000000L,
    )
    let physical = @sparse_index_physical_ir.RetainedIndexPlan::new(semantic)
    assert_eq(physical.published_slots(), expected)
  }
}

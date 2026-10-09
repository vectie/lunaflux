///|
test "projected sparse lowering keeps query and history domains distinct" {
  for queries in [8, 1] {
    let plan = @precision.IndexedSparseAttentionPrecision::new(
      index_head_count=1,
      index_head_dimension=4,
      query_head_count=2,
      key_value_head_count=1,
      query_key_dimension=4,
      value_dimension=4,
      pool_size=4,
      top_k=8,
      append_incomplete_tail=true,
      maximum_queries=queries,
      maximum_history=32,
      maximum_workspace_bytes=100000L,
    )
    let prefix = if queries == 1 { "sparse_decode" } else { "sparse_prefill" }
    let source = @indexed_sparse_attention_execution_cuda_source.indexed_sparse_attention_source(
      plan,
      symbol_prefix=prefix,
    )
    let text = @utf8.decode(source.source_bytes())
    assert_true(text.contains("live_tokens > 32"))
    assert_true(text.contains("query_count > " + queries.to_string()))
    assert_true(text.contains("keys_live > 32"))
    assert_true(
      text.contains(
        "cell < " +
        queries.to_string() +
        "ULL * (unsigned long long)output_width",
      ),
    )
    assert_true(
      text.contains(
        "cell < " + (queries * 8).to_string() + "ULL; ++cell) output[cell]",
      ),
    )
    if @env.get_env_var("LUNA_EXPORT_INDEXED_SPARSE") is Some("1") {
      println("LUNA_SOURCE_BEGIN " + prefix)
      println(text)
      let history = @precision.IndexedSparseHistoryPrecision::new(
        plan,
        maximum_state_bytes=100000L,
      )
      let append = @indexed_sparse_attention_execution_cuda_source.indexed_sparse_history_source(
        history,
        symbol_prefix=prefix + "_history",
      )
      println(@utf8.decode(append.source_bytes()))
      println("LUNA_SOURCE_END")
    }
  }
}

///|
test "learned indexer composes checkpoint-backed full heads and causal compressed selection" {
  for
    model in [
      "deepseek-ai/DeepSeek-V4-Flash-0731", "deepseek-ai/DeepSeek-V4-Pro",
    ] {
    let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(model)
    for layer in 0..<spec.layer_count() {
      if spec.compression_ratio(layer) != Some(4) {
        continue
      }
      let plan = @deepseek_v4_query_kv.index_selection_plan(
        spec,
        layer~,
        maximum_rows=3,
        maximum_context=16,
        maximum_workspace_bytes=1048576L,
      )
      assert_eq(
        plan.geometry(),
        (
          spec.index_head_count(),
          spec.index_head_dimension(),
          4,
          3,
          16,
          spec.index_top_k(),
        ),
      )
      assert_eq(plan.cache_capacity(), 4)
      assert_eq(plan.selected_width(), 4)
      let owner = @deepseek_v4_query_kv.DeepSeekLearnedIndexer::new(
        spec,
        layer~,
        maximum_rows=3,
        maximum_context=16,
        maximum_state_bytes=1048576L,
        maximum_workspace_bytes=1048576L,
      )
      let text = @utf8.decode(owner.source_bytes())
      assert_true(text.contains("_index_queries_queries"))
      assert_true(text.contains("_indexer_compressed_history_publish"))
      assert_true(text.contains("_index_selection_scores"))
      assert_true(text.contains("_index_selection_select"))
      owner.close()
      owner.close()
    }
  }
}

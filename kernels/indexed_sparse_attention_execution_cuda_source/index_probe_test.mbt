///|
test "export retained index paired probe" {
  if @env.get_env_var("LUNA_EXPORT_RETAINED_INDEX") != Some("1") {
    return
  }
  let p = @precision.IndexedSparseAttentionPrecision::new(
    index_head_count=4,
    index_head_dimension=32,
    query_head_count=4,
    key_value_head_count=1,
    query_key_dimension=32,
    value_dimension=32,
    pool_size=4,
    top_k=256,
    append_incomplete_tail=true,
    maximum_queries=16,
    maximum_history=256,
    maximum_workspace_bytes=10000000L,
  )
  let s = @indexed_sparse_attention_execution_cuda_source.indexed_sparse_attention_source(
    p,
    symbol_prefix="bench",
  )
  let old = @index.projected_index_score_stable_topk_bf16_source(
    head_count=4,
    head_dimension=32,
    top_k=256,
    pool_size=4,
    append_incomplete_tail=true,
    maximum_tokens=256,
    maximum_queries=16,
    function_symbol="serial_index",
  )
  println("LUNA_SOURCE_BEGIN")
  println(@utf8.decode(s.source_bytes()))
  println(@utf8.decode(old.source_bytes()))
  println("static constexpr size_t scratch_bytes=\{s.scratch_bytes()}ULL;")
  println("LUNA_SOURCE_END")
}

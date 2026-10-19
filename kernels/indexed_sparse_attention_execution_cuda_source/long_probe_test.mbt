///|
/// Offline export at the deployed long-prefill geometry. Comparison remains
/// outside the compiler; this calls the same public lowering as serving.
test "export long paged sparse probe" {
  if @env.get_env_var("LUNA_LONG_PAGED_EXPORT") is Some(prefix) {
    guard prefix == "old" || prefix == "next" else { fail("probe prefix") }
    let plan = @precision.IndexedSparseAttentionPrecision::new(
      index_head_count=32,
      index_head_dimension=128,
      query_head_count=32,
      key_value_head_count=32,
      query_key_dimension=256,
      value_dimension=256,
      pool_size=4,
      top_k=2048,
      append_incomplete_tail=true,
      maximum_queries=64,
      maximum_history=8192,
      maximum_workspace_bytes=1073741824L,
      attention_accumulation=if @env.get_env_var("LUNA_LONG_MATRIX") ==
        Some("1") {
        TiledFloat32
      } else {
        OrderedFloat32
      },
      paged_history=@history.PagedHistoryDomain::new(
        rows_per_page=16,
        physical_pages=513,
        maximum_context=8192,
        maximum_sequences=1,
      ).unwrap(),
    )
    let source = @indexed_sparse_attention_execution_cuda_source.indexed_sparse_attention_source(
      plan,
      symbol_prefix=prefix,
    )
    println("LUNA_LONG_BEGIN")
    println(
      "static const bool \{prefix}_matrix=\{plan.attention_accumulation() is TiledFloat32};",
    )
    println(@utf8.decode(source.source_bytes()))
    for stage in 0..<3 {
      let (grid, workers) = source.launch(stage)
      println(
        "static const int \{prefix}_grid_\{stage}=\{grid}, \{prefix}_workers_\{stage}=\{workers};",
      )
    }
    println(
      "static const unsigned long long \{prefix}_scratch=\{source.scratch_bytes()}ULL;",
    )
    println("LUNA_LONG_END")
  }
}

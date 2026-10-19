///|
fn paged_fixture(
  paged : Bool,
) -> @precision.IndexedSparseAttentionPrecision raise {
  @precision.IndexedSparseAttentionPrecision::new(
    index_head_count=2,
    index_head_dimension=4,
    query_head_count=2,
    key_value_head_count=1,
    query_key_dimension=4,
    value_dimension=4,
    pool_size=2,
    top_k=4,
    append_incomplete_tail=true,
    maximum_queries=8,
    maximum_history=16,
    maximum_workspace_bytes=100000L,
    paged_history?=if paged {
      @history.PagedHistoryDomain::new(
        rows_per_page=4,
        physical_pages=8,
        maximum_context=16,
        maximum_sequences=4,
      )
    } else {
      None
    },
  )
}

///|
test "paged sparse storage follows physical pool and bindings follow request domain" {
  let plan = paged_fixture(true)
  let history = @precision.IndexedSparseHistoryPrecision::new(
    plan,
    maximum_state_bytes=100000L,
  )
  assert_eq(plan.physical_history(), 32)
  assert_eq(plan.frame_bytes()[0], 128L)
  assert_eq(history.buffer_bytes()[0], 256L)
  assert_eq(history.buffer_bytes()[6], 76L)
  let source = @indexed_sparse_attention_execution_cuda_source.indexed_sparse_attention_source(
    plan,
    symbol_prefix="paged",
  )
  let append = @indexed_sparse_attention_execution_cuda_source.indexed_sparse_history_source(
    history,
    symbol_prefix="ph",
  )
  assert_eq(source.launch(0), (8, 128))
  assert_eq(append.launch(0), (1, 256))
  assert_eq(append.launch(2), (1, 256))
  let text = @utf8.decode(source.source_bytes())
  assert_true(text.contains("const int32_t *row_sequences"))
  assert_true(text.contains("candidate_valid[physical]==page_generations[at]"))
  assert_true(
    text.contains("retained[visible++] = page_ids[first_page+key/4]*4+key%4"),
  )
  assert_false(text.contains("counts[2] != 1"))
  if @env.get_env_var("LUNA_PAGED_SPARSE_EXPORT") == Some("1") {
    let dense = @indexed_sparse_attention_execution_cuda_source.indexed_sparse_attention_source(
      paged_fixture(false),
      symbol_prefix="dense",
    )
    println("LUNA_PAGED_BEGIN")
    println(@utf8.decode(source.source_bytes()))
    println(@utf8.decode(append.source_bytes()))
    println(@utf8.decode(dense.source_bytes()))
    println("LUNA_PAGED_END")
  }
}

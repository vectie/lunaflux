///|
test "query reuse keeps distinct masks and sequence cohorts" {
  let domain = @physical.PagedHistoryDomain::new(
    rows_per_page=8,
    physical_pages=12,
    maximum_context=64,
    maximum_sequences=2,
  ).unwrap()
  let source = @selected_sparse_attention_cuda_source.selected_sparse_source(
    query_heads=4,
    kv_heads=2,
    qk_dimension=32,
    value_dimension=24,
    selected_width=17,
    maximum_queries=31,
    maximum_history=64,
    symbol="reuse_fixture",
    paged_history=domain,
    accumulation=TiledFloat32,
  )
  assert_eq(source.launch(), (124, 256))
  assert_eq(source.scratch_bytes(), 31744L)
  let text = @utf8.decode(source.source_bytes())
  assert_true(text.contains("sequence_id[r]==seq"))
  assert_true(text.contains("atomicOr(&mask"))
  assert_true(text.contains("const int active_tile=any"))
  assert_true(text.contains("__shfl_xor_sync"))
  assert_true(text.contains("__float2bfloat16_rn"))
  assert_true(text.contains("if(counts[3]>=16)reuse_fixture_cohort"))
  assert_true(text.contains("else reuse_fixture_single"))
  // The cohort renderer has its own matrix primitive. No row-zero-only MMA
  // is emitted for incomplete cohorts; their ordered reductions are retained.
  assert_false(text.contains("reuse_fixture_mma("))
  assert_true(text.contains("accumulator = __fadd_rn"))
  if @env.get_env_var("LUNA_QUERY_REUSE_EXPORT") == Some("1") {
    println("LUNA_REUSE_BEGIN")
    println(text)
    println("LUNA_REUSE_END")
  }
}

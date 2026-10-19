///|
test "matrix scoring owns complete head products and retains publication effects" {
  let semantic = @precision.IndexedSparseAttentionPrecision::new(
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
    paged_history=@history.PagedHistoryDomain::new(
      rows_per_page=16,
      physical_pages=513,
      maximum_context=8192,
      maximum_sequences=8,
    ).unwrap(),
    index_scoring=RowScaledE4M3Matrix,
  )
  let plan = @sparse_index_physical_ir.MatrixScorePlan::new(
    semantic,
    candidate_width=32,
  ).unwrap()
  assert_eq(plan.operand_bytes(), (4096L, 4224L))
  assert_eq(plan.tiles().workgroups(), 4096)
  assert_eq(plan.tiles().owner(28, 29), Some((28, 0)))
  assert_eq(plan.tiles().owner(29, 29), Some((0, 32)))
  assert_eq(plan.tiles().owner(29 * 64, 29), None)
  let program = plan.tiles().program()
  assert_eq(program.steps().length(), 2)
  assert_eq(semantic.frame_bytes()[0], 270864L)
}

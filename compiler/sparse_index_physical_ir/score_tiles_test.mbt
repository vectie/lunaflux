///|
test "candidate score partitions retain the producer consumer dependency" {
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
  )
  let index = @sparse_index_physical_ir.RetainedIndexPlan::new(semantic)
  assert_true(index.score_tiles(0) is None)
  assert_true(index.score_tiles(2048) is None)
  let plan = index.score_tiles(128).unwrap()
  assert_eq(plan.geometry(), (64, 2048, 128))
  assert_eq(plan.workgroups(), 1024)
  assert_eq(plan.tiles_per_query(), 16)
  let effects = plan.program()
  assert_true(effects.happens_before(0, 1))
  assert_eq(effects.arenas()[0].byte_capacity(), index.scratch_bytes())
  assert_eq(effects.lower(Ordered).commands(ComputeLane).length(), 2)
  assert_eq(index.score_tiles(127).unwrap().workgroups(), 1088)
  for query in 0..<64 {
    assert_eq(plan.owner(query, 64), Some((query, 0)))
  }
  assert_eq(plan.owner(64, 64), Some((0, 128)))
  assert_eq(plan.owner(15, 1), Some((0, 1920)))
  assert_eq(plan.owner(16, 1), None)
  assert_eq(plan.owner(0, 0), None)
  assert_eq(plan.owner(0, 65), None)
}

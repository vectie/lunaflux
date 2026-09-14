///|
test "generated block-owned matrix domain preserves compatibility geometry" {
  let specs = block_owned_matrix_specs()
  assert_eq(specs.length(), 8)
  let expected = [
    (310, 16, 32, 1, 32),
    (311, 16, 64, 1, 32),
    (312, 32, 32, 1, 32),
    (313, 32, 64, 1, 64),
    (314, 64, 32, 1, 32),
    (315, 64, 64, 1, 64),
    (316, 32, 32, 2, 64),
    (317, 64, 32, 2, 64),
  ]
  for i, spec in specs {
    assert_eq(
      (
        spec.stable_id,
        spec.query_rows,
        spec.key_value_rows,
        spec.pipeline_stages,
        spec.minimum_context_tokens,
      ),
      expected[i],
    )
    assert_eq(spec.phase, Prefill)
    assert_eq(spec.compute, AttentionMatrixQkPv)
    assert_eq(spec.subgroups_per_block, 8)
    assert_eq(spec.vector_bytes, 16)
    assert_eq(spec.minimum_query_rows, spec.query_rows)
    assert_eq(
      spec.memory,
      if i < 6 {
        AttentionStagedShared
      } else {
        AttentionAsyncStagedShared
      },
    )
  }
}

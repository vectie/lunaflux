///|
test "scalar subgroup and matrix-QK domains preserve historical geometry" {
  let specs = attention_candidate_specs()
  let expected = [
    (100, 1, 16, 1, 1, 4, 1),
    (200, 4, 32, 4, 1, 8, 16),
    (210, 8, 64, 4, 1, 8, 32),
    (211, 8, 64, 4, 3, 16, 32),
    (220, 16, 64, 8, 1, 8, 32),
    (221, 16, 128, 8, 3, 16, 32),
    (290, 16, 32, 4, 1, 16, 32),
    (291, 16, 64, 4, 1, 16, 32),
    (300, 16, 64, 8, 1, 16, 32),
    (301, 16, 128, 8, 3, 16, 32),
    (400, 1, 16, 1, 1, 4, 1),
    (410, 1, 64, 4, 2, 8, 32),
    (411, 1, 128, 4, 3, 16, 32),
  ]
  assert_eq(specs.length(), expected.length())
  for i, spec in specs {
    assert_eq(
      (
        spec.stable_id,
        spec.query_rows,
        spec.key_value_rows,
        spec.subgroups_per_block,
        spec.pipeline_stages,
        spec.vector_bytes,
        spec.minimum_context_tokens,
      ),
      expected[i],
    )
    assert_eq(spec.minimum_query_rows, spec.query_rows)
    assert_eq(spec.phase, if i < 10 { Prefill } else { Decode })
    let scalar = spec.stable_id == 100 || spec.stable_id == 400
    assert_eq(
      spec.compute,
      if scalar {
        AttentionScalarDot
      } else if i >= 6 && i < 10 {
        AttentionMatrixDot
      } else {
        AttentionSubgroupDot
      },
    )
    assert_eq(
      spec.memory,
      if scalar {
        AttentionDirectGlobal
      } else if spec.pipeline_stages == 3 {
        AttentionAsyncStagedShared
      } else {
        AttentionStagedShared
      },
    )
  }
}

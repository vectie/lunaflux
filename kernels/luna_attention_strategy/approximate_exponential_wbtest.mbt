///|
test "approximate exponential candidates require independent explicit admission" {
  let make = (allowed, alternative) => {
    AttentionCompileProblem::new(
      shape=compiler_test_shape(),
      phase=Prefill,
      numeric_type=AttentionBFloat16,
      query_rows=128,
      context_tokens=4096,
      causal=true,
      allow_alternative_softmax_reduction=alternative,
      allow_approximate_exponential=allowed,
    )
  }
  let strict = make(false, true)
  let approximate = make(true, true)
  assert_false(strict.allows_approximate_exponential())
  assert_true(approximate.allows_approximate_exponential())
  assert_true(approximate.with_query_rows(64).allows_approximate_exponential())
  let strict_candidates = generate_attention_tile_candidates(
    strict,
    compiler_test_capabilities(),
  )
  let approximate_candidates = generate_attention_tile_candidates(
    approximate,
    compiler_test_capabilities(),
  )
  assert_false(
    strict_candidates.any(c => c.compute() == AttentionMatrixQkPvApproxExp),
  )
  let candidate = compiler_candidate_with_id(approximate_candidates, 30322)
  assert_eq(candidate.compute(), AttentionMatrixQkPvApproxExp)
  let (queries, keys, dimensions) = candidate.tile().dimensions()
  let (subgroups, _, _) = candidate.execution_geometry()
  assert_eq(
    attention_fragment_ownership(
      candidate.compute(),
      query_rows=queries,
      key_rows=keys,
      head_columns=dimensions,
      subgroups~,
    ),
    QueryOwnedFragments,
  )
  assert_eq(
    candidate.tile().dimensions(),
    compiler_candidate_with_id(strict_candidates, 322).tile().dimensions(),
  )
  expect_attention_compiler_rule(() => ignore(make(true, false)), Problem)
  for
    (phase, numeric, query_rows) in [
      (Decode, AttentionBFloat16, 1),
      (Prefill, AttentionFloat32, 128),
    ] {
    expect_attention_compiler_rule(
      () => {
        ignore(
          AttentionCompileProblem::new(
            shape=compiler_test_shape(),
            phase~,
            numeric_type=numeric,
            query_rows~,
            context_tokens=4096,
            causal=true,
            allow_alternative_softmax_reduction=true,
            allow_approximate_exponential=true,
          ),
        )
      },
      Problem,
    )
  }
}

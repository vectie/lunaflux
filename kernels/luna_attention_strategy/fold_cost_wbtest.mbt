///|
test "staging retile does not change ordered subgroup softmax work" {
  let problem = AttentionCompileProblem::new(
    shape=compiler_test_shape(),
    phase=Decode,
    numeric_type=AttentionBFloat16,
    batch_rows=8,
    query_rows=1,
    context_tokens=1528,
    causal=true,
  )
  let candidates = generate_attention_tile_candidates(
    problem,
    compiler_test_capabilities(),
  )
  let synchronous = compiler_candidate_with_id(candidates, 430)
  let narrow = compiler_candidate_with_id(candidates, 441)
  let wide = compiler_candidate_with_id(candidates, 440)
  assert_eq(attention_candidate_online_fold_steps(problem, synchronous), 764)
  assert_eq(attention_candidate_online_fold_steps(problem, narrow), 764)
  assert_eq(attention_candidate_online_fold_steps(problem, wide), 764)
  assert_true(
    attention_candidate_static_cost(problem, narrow) <
    attention_candidate_static_cost(problem, synchronous),
  )
  assert_true(
    attention_candidate_static_cost(problem, wide) <
    attention_candidate_static_cost(problem, synchronous),
  )
}

///|
test "cost decomposition preserves totals and exposes narrower matrix fold work" {
  let problem = AttentionCompileProblem::new(
    shape=compiler_test_shape(),
    phase=Prefill,
    numeric_type=AttentionBFloat16,
    batch_rows=8,
    query_rows=2048,
    context_tokens=4096,
    causal=true,
    allow_alternative_softmax_reduction=true,
  )
  let candidates = generate_attention_tile_candidates(
    problem,
    compiler_test_capabilities(),
  )
  for candidate in candidates {
    let cost = attention_candidate_cost_breakdown(problem, candidate)
    assert_eq(
      cost.total,
      cost.arithmetic +
      cost.memory +
      cost.scheduling +
      cost.fold +
      cost.transfer,
    )
    assert_eq(cost.total, attention_candidate_static_cost(problem, candidate))
  }
  let wide = attention_candidate_cost_breakdown(
    problem,
    compiler_candidate_with_id(candidates, 322),
  )
  let narrow = attention_candidate_cost_breakdown(
    problem,
    compiler_candidate_with_id(candidates, 324),
  )
  assert_eq(narrow.arithmetic, wide.arithmetic)
  assert_eq(narrow.memory, wide.memory)
  assert_eq(narrow.fold, wide.fold * 2L)
  assert_eq(narrow.transfer, wide.transfer * 2L)
}

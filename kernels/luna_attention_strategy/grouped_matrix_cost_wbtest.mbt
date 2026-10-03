///|
test "grouped matrix static arithmetic charges inactive instruction rows" {
  let problem = AttentionCompileProblem::new(
    shape=AttentionShape::new(
      query_heads=16,
      kv_heads=8,
      head_dimension=128,
      tokens_per_page=8,
    ),
    phase=Decode,
    numeric_type=AttentionBFloat16,
    query_rows=1,
    context_tokens=4096,
    causal=true,
    allow_alternative_softmax_reduction=true,
  )
  let candidates = generate_attention_tile_candidates(
    problem,
    compiler_test_capabilities(),
  )
  let matrix = compiler_candidate_with_id(candidates, 482)
  let cost = attention_candidate_cost_breakdown(problem, matrix)
  assert_eq(cost.arithmetic, 4096L * 128L / 8L)
  assert_eq(matrix.tile().dimensions(), (1, 32, 128))
}

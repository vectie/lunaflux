///|
test "partitioned prefill specializes the low-parallelism query tile" {
  let maximum = @luna_attention_strategy.AttentionCompileProblem::new(
    shape=@luna_attention_strategy.AttentionShape::new(
      query_heads=16,
      kv_heads=8,
      head_dimension=128,
      tokens_per_page=8,
    ),
    phase=Prefill,
    numeric_type=AttentionBFloat16,
    query_rows=46,
    context_tokens=40960,
    causal=true,
  )
  let specialized = qwen_partitioned_prefill_problem(maximum, 16)
  assert_eq(specialized.dimensions(), (16, 40960))
  assert_eq(specialized.shape(), maximum.shape())
  assert_eq(specialized.phase(), maximum.phase())
  assert_eq(specialized.numeric_type(), maximum.numeric_type())
  assert_eq(specialized.batch_rows(), maximum.batch_rows())
  assert_eq(specialized.is_causal(), maximum.is_causal())
}

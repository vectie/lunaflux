///|
test "required query ownership excludes exchanged schedules across query buckets" {
  let shape = AttentionShape::new(
    query_heads=8,
    kv_heads=2,
    head_dimension=64,
    tokens_per_page=16,
  )
  let capabilities = AttentionCompilerCapabilities::new(
    subgroup_width=32,
    max_threads_per_block=256,
    max_shared_memory_bytes=100000L,
    max_pipeline_stages=2,
    supports_async_copy=true,
    matrix_shape=Some(
      AttentionMatrixShape::new(rows=16, columns=16, reduction=16),
    ),
  )
  let problem = AttentionCompileProblem::new(
    shape~,
    phase=Prefill,
    numeric_type=AttentionBFloat16,
    query_rows=4096,
    context_tokens=8192,
    causal=true,
    allow_alternative_softmax_reduction=true,
    key_value_read_view=DenseCurrentAndPagedHistory,
    required_fragment_ownership=Some(QueryOwnedFragments),
  )
  for queries in [1, 16, 32, 64, 128, 129, 1528, 4096] {
    let bucket = problem.with_query_rows(queries)
    let candidates = generate_attention_tile_candidates(bucket, capabilities)
    assert_eq(candidates.length(), 12)
    assert_true(
      candidates.all(c => {
        let (q, k, d) = c.tile().dimensions()
        attention_fragment_ownership(
          c.compute(),
          query_rows=q,
          key_rows=k,
          head_columns=d,
          subgroups=c.execution_geometry().0,
        ) ==
        QueryOwnedFragments
      }),
    )
    assert_true(bucket.allows_alternative_softmax_reduction())
    assert_eq(bucket.key_value_read_view(), DenseCurrentAndPagedHistory)
  }
  let narrow = generate_attention_tile_candidates(problem, capabilities).filter(c => {
      c.stable_id() == 324
    },
  )
  guard narrow is [candidate] else { fail("missing asymmetric reduction tile") }
  assert_eq(candidate.tile().dimensions(), (64, 32, 64))
  assert_eq(candidate.execution_geometry(), (4, 1, 16))
  assert_eq(candidate.shared_memory_bytes(), 16400L)
  assert_false(
    generate_attention_tile_candidates(problem, {
      ..capabilities,
      supports_async_copy: false,
    }).any(c => c.stable_id() == 324),
  )
  try
    AttentionCompileProblem::new(
      shape~,
      phase=Prefill,
      numeric_type=AttentionBFloat16,
      query_rows=4096,
      context_tokens=8192,
      causal=true,
      required_fragment_ownership=Some(QueryOwnedFragments),
    )
  catch {
    InvalidAttentionStrategy(Problem) => ()
    _ => fail("unexpected error")
  } noraise {
    _ => fail("ownership must not silently permit numerical change")
  }
}

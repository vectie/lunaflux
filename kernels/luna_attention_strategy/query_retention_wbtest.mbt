///|
test "retention variants are distinct same-law schedules and need measurements" {
  let shape = AttentionShape::new(
    query_heads=16,
    kv_heads=8,
    head_dimension=128,
    tokens_per_page=8,
  )
  let problem = AttentionCompileProblem::new(
    shape~,
    phase=Prefill,
    numeric_type=AttentionBFloat16,
    batch_rows=16,
    query_rows=2048,
    context_tokens=8192,
    causal=true,
    allow_alternative_softmax_reduction=true,
    required_fragment_ownership=Some(QueryOwnedFragments),
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
  let candidates = generate_attention_tile_candidates(problem, capabilities)
  let ids : Map[Int, Bool] = Map([])
  for candidate in candidates {
    assert_false(ids.contains(candidate.stable_id()))
    ids[candidate.stable_id()] = true
  }
  guard candidates.filter(c => c.stable_id() == 322) is [base] else {
    fail("missing base query ownership")
  }
  for (id, retained) in [(10644, 64), (10645, 128)] {
    guard candidates.filter(c => c.stable_id() == id) is [variant] else {
      fail("missing query retention")
    }
    assert_eq(variant.retained_query_dimensions(), retained)
    assert_eq(variant.tile(), base.tile())
    assert_eq(variant.compute(), base.compute())
    assert_eq(variant.memory(), base.memory())
    assert_eq(variant.execution_geometry(), base.execution_geometry())
    assert_eq(variant.shared_memory_bytes(), base.shared_memory_bytes())
    // Without physical resource/timing observations, retaining registers is
    // an alternative, not a fabricated win over the identical reload schedule.
    assert_eq(
      attention_candidate_static_cost(problem, variant),
      attention_candidate_static_cost(problem, base),
    )
  }
  let backend = AttentionBackendId::new("cuda")
  let target = AttentionTargetId::new("fixture")
  assert_eq(
    compile_attention_tile_plan([], backend~, target~, problem~, capabilities~)
    .candidate()
    .retained_query_dimensions(),
    0,
  )
}

///|
test "incremental frontier reuses pure passes and rebinds resource feedback" {
  let request = transfer_frontier_request([])
  let previous = compile_attention_tile_frontier(request)
  let (same, hits) = compile_attention_tile_frontier_incremental(
    request, previous,
  )
  assert_eq(same, previous)
  assert_eq(hits, previous.variants().length())
  let changed = request.with_resource_feedback(feedback_test_budget(), [])
  let (updated, hits) = compile_attention_tile_frontier_incremental(
    changed, previous,
  )
  assert_eq(hits, previous.variants().length())
  assert_eq(updated, compile_attention_tile_frontier(changed))
  // The original value and its embedded request stay immutable.
  assert_eq(previous, compile_attention_tile_frontier(request))
  let changed_shape = {
    ..request,
    problem: request.problem.with_query_rows(256),
  }
  let (different, misses) = compile_attention_tile_frontier_incremental(
    changed_shape, previous,
  )
  assert_eq(misses, 0)
  assert_eq(different, compile_attention_tile_frontier(changed_shape))
  let changed_target = {
    ..request,
    target: @luna_attention_strategy.AttentionTargetId::new("another-target"),
  }
  let (different, misses) = compile_attention_tile_frontier_incremental(
    changed_target, previous,
  )
  assert_eq(misses, 0)
  assert_eq(different, compile_attention_tile_frontier(changed_target))
}

///|
test "incremental compilation invalidates effects capabilities and numerical permission" {
  let base = transfer_frontier_request([])
  let previous = compile_attention_tile_frontier(base)
  let p = base.problem
  let (query_rows, context_tokens) = p.dimensions()
  for
    (causal, alternative, view) in [
      (false, false, p.key_value_read_view()),
      (true, true, p.key_value_read_view()),
      (true, false, DenseCurrentAndPagedHistory),
    ] {
    let problem = @luna_attention_strategy.AttentionCompileProblem::new(
      shape=p.shape(),
      phase=p.phase(),
      numeric_type=p.numeric_type(),
      batch_rows=p.batch_rows(),
      query_rows~,
      context_tokens~,
      causal~,
      allow_alternative_softmax_reduction=alternative,
      key_value_read_view=view,
    )
    assert_false(problem == p)
    let request = { ..base, problem, }
    let (updated, hits) = compile_attention_tile_frontier_incremental(
      request, previous,
    )
    assert_eq(hits, 0)
    assert_eq(updated, compile_attention_tile_frontier(request))
  }
  let request = {
    ..base,
    capabilities: base.capabilities.with_parallel_workgroup_target(72),
  }
  let (updated, hits) = compile_attention_tile_frontier_incremental(
    request, previous,
  )
  assert_eq(hits, 0)
  assert_eq(updated, compile_attention_tile_frontier(request))
}

///|
test "incremental selection consumes new observations and does not retain old winner" {
  let base = transfer_frontier_request([])
  let previous = compile_attention_tile_frontier(base)
  let candidate = previous
    .variants()
    .filter(v => v.plan().candidate().stable_id() == 316)[0]
    .plan()
    .candidate()
  let record = @luna_attention_strategy.AttentionAutotuneRecord::new(
    key=@luna_attention_strategy.AttentionAutotuneKey::new(
      backend=base.backend,
      target=base.target,
      problem=base.problem,
      candidate~,
    ),
    median_latency_ns=100L,
    sample_count=7,
  )
  let request = transfer_frontier_request([record])
  let (measured, reused) = compile_attention_tile_frontier_incremental(
    request, previous,
  )
  assert_eq(measured, compile_attention_tile_frontier(request))
  assert_eq(reused, 1)
  assert_eq(measured.selected().plan().candidate().stable_id(), 316)
  assert_eq(measured.selected().plan().source(), AttentionOfflineAutotune)
  let (restored, reused) = compile_attention_tile_frontier_incremental(
    base, measured,
  )
  assert_eq(restored, previous)
  assert_eq(reused, 1)
}

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

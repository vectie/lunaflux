///|
test "large legal batch geometry compiles without 32-bit workgroup overflow" {
  let base = functional_compiler_request([], query_rows=4096)
  for batch_rows in [524288, 1048575, 1048576] {
    let problem = @luna_attention_strategy.AttentionCompileProblem::new(
      shape=base.problem.shape(),
      phase=Prefill,
      numeric_type=AttentionBFloat16,
      batch_rows~,
      query_rows=4096,
      context_tokens=4096,
      causal=true,
    )
    let request = { ..base, problem, }
    let frontier = compile_attention_tile_frontier(request)
    assert_true(frontier.variants().length() > 0)
    for variant in frontier.variants() {
      assert_eq(variant.schedule().key_value_partitions(), 1)
      assert_eq(
        variant.optimization().reduction_topology(),
        AttentionOrderedOnlineSoftmaxFold,
      )
      let shared = variant.schedule().execution_geometry().3
      let residency = if shared <= 0L {
        1L
      } else {
        Int64::max(1L, Int64::min(2L, 100000L / shared))
      }
      let cost = variant.plan().static_cost()
      assert_eq(
        attention_tile_resource_score(request, variant),
        1L + (cost - 1L) / residency,
      )
    }
  }
}

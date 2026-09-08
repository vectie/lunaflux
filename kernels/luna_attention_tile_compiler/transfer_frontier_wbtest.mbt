///|
fn transfer_frontier_request(
  records : ArrayView[@luna_attention_strategy.AttentionAutotuneRecord],
) -> AttentionTileCompilerRequest raise {
  let base = functional_compiler_request([])
  AttentionTileCompilerRequest::new(
    backend=base.backend,
    target=base.target,
    problem=base.problem,
    capabilities=@luna_attention_strategy.AttentionCompilerCapabilities::new(
      subgroup_width=32,
      max_threads_per_block=256,
      max_shared_memory_bytes=100000L,
      max_pipeline_stages=2,
      supports_async_copy=true,
      matrix_shape=Some(
        @luna_attention_strategy.AttentionMatrixShape::new(
          rows=16,
          columns=16,
          reduction=16,
        ),
      ),
    ),
    records,
  )
}

///|
test "transfer frontier retains overlap alternatives without selecting them blindly" {
  let request = transfer_frontier_request([])
  let frontier = compile_attention_tile_frontier(request)
  let ids = frontier
    .variants()
    .map(variant => variant.plan().candidate().stable_id())
  for id in [312, 314, 316, 317] {
    assert_true(ids.contains(id))
  }
  // Hardware availability is not a performance observation. The conservative
  // resource-feedback choice stays synchronous at this high-query shape.
  assert_eq(frontier.selected().plan().candidate().stable_id(), 313)
  assert_eq(frontier.selected().plan().source(), AttentionStaticCostModel)
  assert_eq(frontier, compile_attention_tile_frontier(request))
  let unsupported = compile_attention_tile_frontier(
    functional_compiler_request([]),
  )
  assert_false(
    unsupported
    .variants()
    .any(variant => {
      variant.plan().candidate().memory() == AttentionAsyncStagedShared
    }),
  )
}

///|
test "exact measured overlap winner still collapses the transfer frontier" {
  let base = transfer_frontier_request([])
  let candidates = @luna_attention_strategy.generate_attention_tile_candidates(
    base.problem,
    base.capabilities,
  )
  guard candidates.filter(candidate => candidate.stable_id() == 316)
    is [candidate] else {
    fail("overlap candidate absent")
  }
  // Synthetic selector fixture only; no latency is installed in an exporter.
  let record = @luna_attention_strategy.AttentionAutotuneRecord::new(
    key=@luna_attention_strategy.AttentionAutotuneKey::new(
      backend=base.backend,
      target=base.target,
      problem=base.problem,
      candidate~,
    ),
    median_latency_ns=100L,
    sample_count=9,
  )
  let frontier = compile_attention_tile_frontier(
    transfer_frontier_request([record]),
  )
  assert_eq(frontier.variants().length(), 1)
  assert_eq(frontier.selected().plan().candidate().stable_id(), 316)
  assert_eq(frontier.selected().plan().source(), AttentionOfflineAutotune)
}

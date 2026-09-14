///|
/// Compiler CPU benchmark, not inference or kernel timing. The same immutable
/// request and output checks are used in both arms, with alternating order.
fn main raise {
  let request = @compiler.AttentionTileCompilerRequest::new(
    backend=@strategy.AttentionBackendId::new("benchmark"),
    target=@strategy.AttentionTargetId::new("portable-matrix16"),
    problem=@strategy.AttentionCompileProblem::new(
      shape=@strategy.AttentionShape::new(
        query_heads=16,
        kv_heads=8,
        head_dimension=128,
        tokens_per_page=8,
      ),
      phase=Prefill,
      numeric_type=AttentionBFloat16,
      query_rows=2048,
      context_tokens=4096,
      causal=true,
      allow_alternative_softmax_reduction=true,
      key_value_read_view=DenseCurrentAndPagedHistory,
      required_fragment_ownership=Some(QueryOwnedFragments),
    ),
    capabilities=@strategy.AttentionCompilerCapabilities::new(
      subgroup_width=32,
      max_threads_per_block=256,
      max_shared_memory_bytes=100000L,
      max_pipeline_stages=2,
      supports_async_copy=true,
      matrix_shape=Some(
        @strategy.AttentionMatrixShape::new(rows=16, columns=16, reduction=16),
      ),
    ),
    [],
  )
  let previous = @compiler.compile_attention_tile_frontier(request)
  let run = fn(incremental : Bool) -> Double raise {
    let start = @bench.monotonic_clock_start()
    for _ in 0..<20 {
      let result = if incremental {
        let (result, hits) = @compiler.compile_attention_tile_frontier_incremental(
          request, previous,
        )
        assert_eq(hits, previous.variants().length())
        result
      } else {
        @compiler.compile_attention_tile_frontier(request)
      }
      assert_eq(result, previous)
    }
    @bench.monotonic_clock_end(start) / 20.0
  }
  ignore(run(false))
  ignore(run(true))
  for trial in 0..<5 {
    let (fresh, reused) = if trial % 2 == 0 {
      let fresh = run(false)
      (fresh, run(true))
    } else {
      let reused = run(true)
      (run(false), reused)
    }
    println(
      "trial=\{trial} candidates=\{previous.variants().length()} fresh_us=\{fresh} reused_us=\{reused}",
    )
  }
}

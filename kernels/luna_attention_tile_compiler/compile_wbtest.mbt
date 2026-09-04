///|
fn functional_compiler_request(
  records : ArrayView[@luna_attention_strategy.AttentionAutotuneRecord],
  query_rows? : Int = 128,
) -> AttentionTileCompilerRequest raise {
  let shape = @luna_attention_strategy.AttentionShape::new(
    query_heads=16,
    kv_heads=4,
    head_dimension=128,
    tokens_per_page=16,
  )
  AttentionTileCompilerRequest::new(
    backend=@luna_attention_strategy.AttentionBackendId::new("generic-gpu"),
    target=@luna_attention_strategy.AttentionTargetId::new("matrix16"),
    problem=@luna_attention_strategy.AttentionCompileProblem::new(
      shape~,
      phase=Prefill,
      numeric_type=AttentionBFloat16,
      query_rows~,
      context_tokens=4096,
      causal=true,
    ),
    capabilities=@luna_attention_strategy.AttentionCompilerCapabilities::new(
      subgroup_width=32,
      max_threads_per_block=256,
      max_shared_memory_bytes=100000L,
      max_pipeline_stages=2,
      supports_async_copy=false,
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
test "functional attention compiler composes typed immutable passes" {
  let request = functional_compiler_request([])
  let selected = select_attention_tile(request)
  let semantic = elaborate_attention_tile(selected)
  let optimized = optimize_attention_tile(semantic)
  let scheduled = schedule_attention_tile(optimized)
  let composed = compile_attention_tile(request)
  assert_eq(selected.plan().candidate().stable_id(), 314)
  assert_eq(composed.plan().candidate().stable_id(), 312)
  assert_eq(semantic.plan(), selected.plan())
  assert_eq(optimized.plan(), selected.plan())
  assert_eq(optimized.optimization().regions().length(), 4)
  assert_false(scheduled == composed)
  assert_eq(scheduled.program(), semantic.program())
  assert_eq(composed.schedule().execution_geometry().3, 45056L)
  assert_eq(scheduled.schedule().program_digest(), semantic.program().digest())
  assert_eq(scheduled.optimization(), optimized.optimization())
  assert_true(scheduled.schedule().hoists_paged_key_value_page_lookup())
  assert_true(scheduled.schedule().reuses_online_softmax_tile_storage())
  assert_eq(scheduled.canonical_bytes().length() > 0, true)
  assert_eq(scheduled.digest().as_hex().length(), 64)
}

///|
test "offline autotune remains authoritative over resource feedback" {
  let base = functional_compiler_request([])
  let problem = base.problem()
  let candidates = @luna_attention_strategy.generate_attention_tile_candidates(
    problem,
    base.capabilities,
  )
  guard candidates.filter(candidate => candidate.stable_id() == 310)
    is [candidate] else {
    fail("autotune candidate unavailable")
  }
  let record = @luna_attention_strategy.AttentionAutotuneRecord::new(
    key=@luna_attention_strategy.AttentionAutotuneKey::new(
      backend=base.backend,
      target=base.target,
      problem~,
      candidate~,
    ),
    median_latency_ns=100L,
    sample_count=7,
  )
  let compiled = compile_attention_tile(functional_compiler_request([record]))
  assert_eq(compiled.plan().source(), AttentionOfflineAutotune)
  assert_eq(compiled.plan().candidate().stable_id(), 310)
}

///|
test "functional attention compiler is referentially transparent" {
  let request = functional_compiler_request([])
  let first = compile_attention_tile(request)
  let second = compile_attention_tile(request)
  assert_eq(first, second)
  assert_eq(first.canonical_bytes(), second.canonical_bytes())
  assert_eq(first.digest(), second.digest())
}

///|
test "partitioned compiler path materializes optimizer reduction topology" {
  let request = functional_compiler_request([], query_rows=64)
  let first = compile_partitioned_attention_tile(request)
  let repeated = compile_partitioned_attention_tile(request)
  assert_eq(first, repeated)
  assert_eq(first.schedule().key_value_partitions(), 2)
  assert_eq(first.schedule().loops().length(), 6)
  assert_eq(
    first.optimization().reduction_topology(),
    AttentionPartitionedOnlineSoftmaxFold(2),
  )
  assert_false(first.digest() == compile_attention_tile(request).digest())
}

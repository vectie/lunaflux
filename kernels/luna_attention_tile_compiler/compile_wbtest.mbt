///|
fn functional_compiler_request(
  records : ArrayView[@luna_attention_strategy.AttentionAutotuneRecord],
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
      query_rows=128,
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
  assert_eq(semantic.plan(), selected.plan())
  assert_eq(optimized.plan(), selected.plan())
  assert_eq(optimized.optimization().regions().length(), 3)
  assert_eq(scheduled, composed)
  assert_eq(scheduled.program(), semantic.program())
  assert_eq(scheduled.schedule().program_digest(), semantic.program().digest())
  assert_eq(scheduled.optimization(), optimized.optimization())
  assert_eq(scheduled.canonical_bytes().length() > 0, true)
  assert_eq(scheduled.digest().as_hex().length(), 64)
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
  let request = functional_compiler_request([])
  let first = compile_partitioned_attention_tile(request)
  let repeated = compile_partitioned_attention_tile(request)
  assert_eq(first, repeated)
  assert_eq(first.schedule().key_value_partitions(), 8)
  assert_eq(first.schedule().loops().length(), 6)
  assert_eq(
    first.optimization().reduction_topology(),
    @luna_attention_tile_optimizer.AttentionPartitionedOnlineSoftmaxFold(8),
  )
  assert_false(first.digest() == compile_attention_tile(request).digest())
}

///|
fn optimization_program(
  causal : Bool,
  context_tokens? : Int = 4096,
) -> @luna_attention_tile_ir.AttentionTileProgram raise {
  let shape = @luna_attention_strategy.AttentionShape::new(
    query_heads=16,
    kv_heads=4,
    head_dimension=128,
    tokens_per_page=16,
  )
  let problem = @luna_attention_strategy.AttentionCompileProblem::new(
    shape~,
    phase=Prefill,
    numeric_type=AttentionBFloat16,
    query_rows=128,
    context_tokens~,
    causal~,
  )
  let plan = @luna_attention_strategy.compile_attention_tile_plan(
    [],
    backend=@luna_attention_strategy.AttentionBackendId::new("generic-gpu"),
    target=@luna_attention_strategy.AttentionTargetId::new("matrix16"),
    problem~,
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
  )
  @luna_attention_tile_ir.compile_attention_tile_ir(plan)
}

///|
test "optimizer extracts deterministic fused functional regions" {
  let program = optimization_program(true)
  let first = optimize_attention_tile_program(program)
  let second = optimize_attention_tile_program(program)
  assert_eq(first, second)
  assert_eq(first.program_digest(), program.digest())
  assert_eq(first.regions().length(), 3)
  assert_eq(first.regions()[0].rule(), FuseScoreTransform)
  assert_eq(first.regions()[0].members().length(), 3)
  assert_eq(first.regions()[1].rule(), FuseOnlineSoftmaxFold)
  assert_eq(first.regions()[1].members().length(), 3)
  assert_eq(first.regions()[2].rule(), PartitionOnlineSoftmaxFold)
  assert_eq(first.regions()[2].members().length(), 3)
  assert_eq(
    first.reduction_topology(),
    AttentionPartitionedOnlineSoftmaxFold(8),
  )
  assert_eq(first.trace().length(), 3)
  assert_eq(first.trace()[0].changed(), true)
  assert_eq(first.trace()[1].region_count(), 2)
  assert_eq(first.trace()[2].region_count(), 3)
  assert_eq(first.digest().as_hex().length(), 64)
}

///|
test "score fusion is conditional on causal masking" {
  let causal = optimize_attention_tile_program(optimization_program(true))
  let noncausal = optimize_attention_tile_program(optimization_program(false))
  assert_eq(causal.regions()[0].members().length(), 3)
  assert_eq(noncausal.regions()[0].members().length(), 2)
  assert_eq(causal.digest() == noncausal.digest(), false)
}

///|
test "short prefill keeps one ordered online-softmax fold" {
  let optimized = optimize_attention_tile_program(
    optimization_program(true, context_tokens=512),
  )
  assert_eq(optimized.reduction_topology(), AttentionOrderedOnlineSoftmaxFold)
  assert_eq(optimized.regions().length(), 2)
  assert_eq(optimized.trace().length(), 3)
  assert_false(optimized.trace()[2].changed())
}

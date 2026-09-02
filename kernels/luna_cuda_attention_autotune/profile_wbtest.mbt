///|
fn sm120_profile_problem() -> @luna_attention_strategy.AttentionCompileProblem raise {
  @luna_attention_strategy.AttentionCompileProblem::new(
    shape=@luna_attention_strategy.AttentionShape::new(
      query_heads=16,
      kv_heads=4,
      head_dimension=128,
      tokens_per_page=16,
    ),
    phase=Prefill,
    numeric_type=AttentionBFloat16,
    query_rows=128,
    context_tokens=4096,
    causal=true,
  )
}

///|
fn sm120_profile_capabilities() -> @luna_attention_strategy.AttentionCompilerCapabilities raise {
  @luna_attention_strategy.AttentionCompilerCapabilities::new(
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
  )
}

///|
test "sm120 profile replays immutable measurements into generic selection" {
  let problem = sm120_profile_problem()
  let capabilities = sm120_profile_capabilities()
  let records = cuda_sm120_rtx5060ti_prefill_records(problem, capabilities)
  assert_eq(records.length(), 5)
  let first = @luna_attention_strategy.compile_attention_tile_plan(
    records,
    backend=cuda_sm120_rtx5060ti_backend(),
    target=cuda_sm120_rtx5060ti_target(),
    problem~,
    capabilities~,
  )
  let repeated = @luna_attention_strategy.compile_attention_tile_plan(
    records,
    backend=cuda_sm120_rtx5060ti_backend(),
    target=cuda_sm120_rtx5060ti_target(),
    problem~,
    capabilities~,
  )
  assert_eq(first, repeated)
  assert_eq(first.source(), AttentionOfflineAutotune)
  assert_eq(first.candidate().stable_id(), 310)
}

///|
test "sm120 profile remains absent for a different functional problem" {
  let capabilities = sm120_profile_capabilities()
  let decode = @luna_attention_strategy.AttentionCompileProblem::new(
    shape=@luna_attention_strategy.AttentionShape::new(
      query_heads=16,
      kv_heads=4,
      head_dimension=128,
      tokens_per_page=16,
    ),
    phase=Decode,
    numeric_type=AttentionBFloat16,
    query_rows=1,
    context_tokens=4096,
    causal=true,
  )
  assert_eq(
    cuda_sm120_rtx5060ti_prefill_records(decode, capabilities).length(),
    0,
  )
}

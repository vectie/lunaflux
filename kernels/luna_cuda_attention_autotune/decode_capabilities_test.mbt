///|
test "decode occupancy policy reaches the asynchronous compiler schedule" {
  let capabilities = @luna_cuda_attention_autotune.cuda_sm120_decode_compiler_capabilities()
  assert_true(capabilities.supports_async_copy())
  assert_eq(capabilities.limits(), (256, 49152L, 2))
  for batch in [1, 8, 32] {
    for context in [59, 128, 512, 1528, 4096, 40960] {
      let problem = @luna_attention_strategy.AttentionCompileProblem::new(
        shape=@luna_attention_strategy.AttentionShape::new(
          query_heads=16,
          kv_heads=8,
          head_dimension=128,
          tokens_per_page=8,
        ),
        phase=Decode,
        numeric_type=AttentionBFloat16,
        batch_rows=batch,
        query_rows=1,
        context_tokens=context,
        causal=true,
      )
      let compiled = @luna_attention_strategy.compile_attention_tile_plan(
        [],
        backend=@luna_cuda_attention_autotune.cuda_sm120_rtx5060ti_backend(),
        target=@luna_cuda_attention_autotune.cuda_sm120_rtx5060ti_target(),
        problem~,
        capabilities~,
      )
      assert_eq(compiled.candidate().stable_id(), 441)
      assert_eq(compiled.candidate().memory(), AttentionAsyncStagedShared)
      assert_true(compiled.candidate().shared_memory_bytes() <= 49152L)
      assert_eq(compiled.source(), AttentionStaticCostModel)
    }
  }
}

///|
test "available async schedules survive the complete AOT frontier" {
  let problem = @luna_attention_strategy.AttentionCompileProblem::new(
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
  let frontier = compile_cuda_attention_tile_aot_frontier(
    CudaAttentionTileAotRequest::new(
      backend=@luna_attention_strategy.AttentionBackendId::new("cuda"),
      target_id=@luna_attention_strategy.AttentionTargetId::new("sm120-fixture"),
      problem~,
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
      device_target=@catalog.DeviceTarget::new(
        compute_major=12,
        compute_minor=0,
        supports_bf16=true,
        supports_cublas_lt=false,
      ),
      maximum_shared_memory_bytes=100000L,
      abi=@luna_cuda_attention_tile_source.CudaPagedAttentionTileAbi::new(
        query_heads=16,
        key_value_heads=4,
        head_dimension=128,
        tokens_per_page=16,
        input_row_width=3072,
        page_stride_values=8192L,
        physical_page_capacity=16L,
        maximum_query_rows=32,
        maximum_query_tokens=128,
        maximum_page_entries=256,
      ),
      symbol="prefill_transfer",
      maximum_source_bytes=1048576,
    ),
    [],
  )
  assert_eq(
    frontier.selected().functional().plan().candidate().stable_id(),
    312,
  )
  assert_eq(frontier.selected().source().symbol(), "prefill_transfer")
  for id in [316, 317] {
    guard frontier
      .variants()
      .filter(variant => {
        variant.functional().plan().candidate().stable_id() == id
      })
      is [variant] else {
      fail("available transfer schedule was erased before AOT export")
    }
    assert_eq(variant.source().symbol(), "prefill_transfer_c\{id}")
    assert_true(@utf8.decode(variant.source().bytes()).contains("cp.async"))
  }
}

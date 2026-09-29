///|
test "alternative blockwise decode reaches every AOT frontier and distinct symbol" {
  let problem = @luna_attention_strategy.AttentionCompileProblem::new(
    shape=@luna_attention_strategy.AttentionShape::new(
      query_heads=16,
      kv_heads=4,
      head_dimension=128,
      tokens_per_page=16,
    ),
    phase=Decode,
    numeric_type=AttentionBFloat16,
    query_rows=1,
    batch_rows=32,
    context_tokens=4096,
    causal=true,
    allow_alternative_softmax_reduction=true,
  )
  let request = CudaAttentionTileAotRequest::new(
    backend=@luna_attention_strategy.AttentionBackendId::new("cuda"),
    target_id=@luna_attention_strategy.AttentionTargetId::new("sm120-fixture"),
    problem~,
    capabilities=@luna_attention_strategy.AttentionCompilerCapabilities::new(
      subgroup_width=32,
      max_threads_per_block=256,
      max_shared_memory_bytes=49152L,
      max_pipeline_stages=2,
      supports_async_copy=true,
    ),
    device_target=@catalog.DeviceTarget::new(
      compute_major=12,
      compute_minor=0,
      supports_bf16=true,
      supports_cublas_lt=false,
    ),
    maximum_shared_memory_bytes=49152L,
    abi=@luna_cuda_attention_tile_source.CudaPagedAttentionTileAbi::new(
      query_heads=16,
      key_value_heads=4,
      head_dimension=128,
      tokens_per_page=16,
      input_row_width=3072,
      page_stride_values=8192L,
      physical_page_capacity=256L,
      maximum_query_rows=32,
      maximum_query_tokens=128,
      maximum_page_entries=256,
    ),
    symbol="decode",
    maximum_source_bytes=1048576,
  )
  let frontier = compile_cuda_attention_tile_aot_frontier(request, [])
  for id in [450, 451] {
    guard frontier
      .variants()
      .filter(v => v.functional().plan().candidate().stable_id() == id)
      is [variant] else {
      fail("blockwise decode was not emitted")
    }
    assert_true(variant.source().symbol().has_suffix("_blockwise_f32_v1"))
    assert_true(
      variant.lowering().physical_program() is BlockwiseKeyProgram(_, _),
    )
    assert_true(
      @utf8.decode(variant.functional().program().canonical_bytes()).contains(
        "blockwise-f32-probability-v1",
      ),
    )
  }
  assert_eq(
    frontier,
    compile_cuda_attention_tile_aot_frontier(request, [], previous=frontier),
  )
}

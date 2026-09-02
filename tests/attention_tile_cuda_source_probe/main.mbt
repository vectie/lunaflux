///|
fn main raise {
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
    context_tokens=4096,
    causal=true,
  )
  let capabilities = @luna_attention_strategy.AttentionCompilerCapabilities::new(
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
  let compiled = @luna_attention_tile_compiler.compile_attention_tile(
    @luna_attention_tile_compiler.AttentionTileCompilerRequest::new(
      backend=@luna_attention_strategy.AttentionBackendId::new("cuda"),
      target=@luna_attention_strategy.AttentionTargetId::new("sm120-probe"),
      problem~,
      capabilities~,
      [],
    ),
  )
  let lowering = @luna_cuda_attention_tile_lowering.lower_attention_tile_schedule_to_cuda(
    compiled.program(),
    compiled.schedule(),
    @catalog.DeviceTarget::new(
      compute_major=12,
      compute_minor=0,
      supports_bf16=true,
      supports_cublas_lt=false,
    ),
    100000L,
  )
  let source = @luna_cuda_attention_tile_source.emit_cuda_attention_tile_source(
    lowering,
    @luna_cuda_attention_tile_source.CudaPagedAttentionTileAbi::new(
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
    "lunaflux_attention_prefill_tile_v1",
    1048576,
  )
  println(@utf8.decode(source.bytes()))
}

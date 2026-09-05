///|
/// Test-only schedule selection, never an observed autotune record.
fn emit_decode_probe(grouped : Bool, partitioned : Bool) -> Unit raise {
  let shape = @luna_attention_strategy.AttentionShape::new(
    query_heads=16,
    kv_heads=8,
    head_dimension=128,
    tokens_per_page=16,
  )
  let problem = @luna_attention_strategy.AttentionCompileProblem::new(
    shape~,
    phase=Decode,
    numeric_type=AttentionBFloat16,
    query_rows=1,
    batch_rows=1,
    context_tokens=4096,
    causal=true,
  )
  let capabilities = @luna_attention_strategy.AttentionCompilerCapabilities::new(
    subgroup_width=32,
    max_threads_per_block=if grouped { 256 } else { 32 },
    max_shared_memory_bytes=if grouped { 100000L } else { 1L },
    max_pipeline_stages=1,
    supports_async_copy=false,
    parallel_workgroup_target=128,
  )
  let backend = @luna_attention_strategy.AttentionBackendId::new("cuda")
  let target_id = @luna_attention_strategy.AttentionTargetId::new(
    "decode-probe",
  )
  let device_target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
  let abi = @luna_cuda_attention_tile_source.CudaPagedAttentionTileAbi::new(
    query_heads=16,
    key_value_heads=8,
    head_dimension=128,
    tokens_per_page=16,
    input_row_width=4096,
    page_stride_values=16384L,
    physical_page_capacity=8192L,
    maximum_query_rows=32,
    maximum_query_tokens=128,
    maximum_page_entries=8192,
  )
  if partitioned {
    let compiled = @luna_cuda_attention_tile_aot.compile_cuda_partitioned_attention_tile_aot(
      @luna_cuda_attention_tile_aot.CudaPartitionedAttentionTileAotRequest::new(
        backend~,
        target_id~,
        problem~,
        capabilities~,
        device_target~,
        maximum_shared_memory_bytes=100000L,
        abi~,
        partial_symbol="decode_partial",
        merge_symbol="decode_merge",
        maximum_workspace_bytes=67108864L,
        maximum_source_bytes=1048576,
      ),
      [],
    )
    let (x, y, threads, shared) = compiled.lowering().launch()
    println(
      "// launch=\{x},\{y},\{threads},\{shared} partitions=" +
      compiled.lowering().key_value_partitions().to_string(),
    )
    println(@utf8.decode(compiled.source().bytes()))
  } else {
    let compiled = @luna_cuda_attention_tile_aot.compile_cuda_attention_tile_aot(
      @luna_cuda_attention_tile_aot.CudaAttentionTileAotRequest::new(
        backend~,
        target_id~,
        problem~,
        capabilities~,
        device_target~,
        maximum_shared_memory_bytes=100000L,
        abi~,
        symbol="decode_baseline",
        maximum_source_bytes=1048576,
      ),
      [],
    )
    let (x, y, threads, shared) = compiled.lowering().launch()
    println("// launch=\{x},\{y},\{threads},\{shared}")
    println(@utf8.decode(compiled.source().bytes()))
  }
}

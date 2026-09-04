///|
test "CUDA AOT compiler purely composes functional and device stages" {
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
  let backend = @luna_attention_strategy.AttentionBackendId::new("cuda")
  let target_id = @luna_attention_strategy.AttentionTargetId::new(
    "sm120-fixture",
  )
  let candidates = @luna_attention_strategy.generate_attention_tile_candidates(
    problem, capabilities,
  )
  guard candidates.filter(candidate => candidate.stable_id() == 310)
    is [candidate] else {
    fail("dual-MMA candidate unavailable")
  }
  let records = [
    @luna_attention_strategy.AttentionAutotuneRecord::new(
      key=@luna_attention_strategy.AttentionAutotuneKey::new(
        backend~,
        target=target_id,
        problem~,
        candidate~,
      ),
      median_latency_ns=31048L,
      sample_count=100,
    ),
  ]
  let request = CudaAttentionTileAotRequest::new(
    backend~,
    target_id~,
    problem~,
    capabilities~,
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
    symbol="lunaflux_attention_prefill_tile_v1",
    maximum_source_bytes=1048576,
  )
  let first = compile_cuda_attention_tile_aot(request, records)
  let repeated = compile_cuda_attention_tile_aot(request, records)
  assert_eq(first, repeated)
  assert_eq(first.functional().plan().candidate().stable_id(), 310)
  assert_eq(first.lowering().launch(), (128, 16, 256, 30720L))
  assert_true(
    @utf8.decode(first.source().bytes()).contains(
      "wmma::load_matrix_sync(probability_fragment",
    ),
  )
  assert_eq(first.digest().as_hex().length(), 64)
}

///|
test "partitioned CUDA AOT compiler emits two-entry-point prefill pipeline" {
  let problem = @luna_attention_strategy.AttentionCompileProblem::new(
    shape=@luna_attention_strategy.AttentionShape::new(
      query_heads=16,
      kv_heads=4,
      head_dimension=128,
      tokens_per_page=16,
    ),
    phase=Prefill,
    numeric_type=AttentionBFloat16,
    query_rows=64,
    context_tokens=4096,
    causal=true,
  )
  let compilation = compile_cuda_partitioned_attention_tile_aot(
    CudaPartitionedAttentionTileAotRequest::new(
      backend=@luna_attention_strategy.AttentionBackendId::new("cuda"),
      target_id=@luna_attention_strategy.AttentionTargetId::new("sm120-fixture"),
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
      partial_symbol="lunaflux_attention_prefill_partial_v1",
      merge_symbol="lunaflux_attention_prefill_merge_v1",
      maximum_workspace_bytes=16777216L,
      maximum_source_bytes=1048576,
    ),
    [],
  )
  assert_eq(compilation.functional().schedule().key_value_partitions(), 2)
  assert_eq(compilation.lowering().key_value_partitions(), 2)
  assert_eq(
    compilation.source().symbols().0,
    "lunaflux_attention_prefill_partial_v1",
  )
  assert_eq(
    compilation.source().symbols().1,
    "lunaflux_attention_prefill_merge_v1",
  )
  assert_eq(compilation.source().workspace().total_bytes(), 2129920L)
  assert_eq(compilation.digest().as_hex().length(), 64)
}

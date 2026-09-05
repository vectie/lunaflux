///|
/// Offline ABI adapter for the portable partition-map/fold/merge compiler.
/// The selected shared-KV schedule comes from the portable cost model. The
/// adapter asserts its geometry against the independently bound runtime ABI.
fn compile_readonly_decode_split(
  shape : @luna_attention_strategy.AttentionShape,
  profile : @launch_contract.PagedKernelProfile,
  target : @catalog.DeviceTarget,
  page_stride_bytes : Int64,
  physical_page_capacity : Int64,
  maximum_context_tokens : Int,
  workspace : @luna_attention_strategy.AttentionDecodeWorkspacePlan,
  partial_symbol : String,
  merge_symbol : String,
  maximum_source_bytes : Int,
) -> @luna_cuda_attention_tile_aot.CudaPartitionedAttentionTileAotCompilation raise PagedAttentionAotError {
  let (query_heads, kv_heads, head_dimension, tokens_per_page) = shape.dimensions()
  let problem = @luna_attention_strategy.AttentionCompileProblem::new(
    shape~,
    phase=Decode,
    numeric_type=AttentionBFloat16,
    query_rows=1,
    batch_rows=1,
    context_tokens=maximum_context_tokens,
    causal=true,
  ) catch {
    _ => raise Invalid(Shape)
  }
  let capabilities = @luna_attention_strategy.AttentionCompilerCapabilities::new(
    subgroup_width=32,
    max_threads_per_block=256,
    max_shared_memory_bytes=100000L,
    max_pipeline_stages=1,
    supports_async_copy=false,
    matrix_shape=None,
    parallel_workgroup_target=query_heads * workspace.split_count(),
  ) catch {
    _ => raise Invalid(Dimensions)
  }
  let abi = @luna_cuda_attention_tile_source.CudaPagedAttentionTileAbi::new(
    query_heads~,
    key_value_heads=kv_heads,
    head_dimension~,
    tokens_per_page~,
    input_row_width=(query_heads + 2 * kv_heads) * head_dimension,
    page_stride_values=page_stride_bytes / 2L,
    physical_page_capacity~,
    maximum_query_rows=profile.max_query_rows(),
    maximum_query_tokens=profile.max_query_tokens(),
    maximum_page_entries=profile.max_page_table_entries(),
  ) catch {
    _ => raise Invalid(Profile)
  }
  let compiled = @luna_cuda_attention_tile_aot.compile_cuda_partitioned_attention_tile_aot(
    @luna_cuda_attention_tile_aot.CudaPartitionedAttentionTileAotRequest::new(
      backend=@luna_attention_strategy.AttentionBackendId::new("cuda"),
      target_id=@luna_attention_strategy.AttentionTargetId::new(
        "shared-kv-decode-v2",
      ),
      problem~,
      capabilities~,
      device_target=target,
      maximum_shared_memory_bytes=100000L,
      abi~,
      partial_symbol~,
      merge_symbol~,
      maximum_workspace_bytes=workspace.total_bytes(),
      maximum_source_bytes~,
    ),
    [],
  ) catch {
    _ => raise Unsupported(Profile)
  }
  if compiled.lowering().key_value_partitions() != workspace.split_count() ||
    compiled.source().workspace().total_bytes() != workspace.total_bytes() ||
    compiled.source().workspace().region_offsets_bytes() !=
    workspace.region_offsets_bytes() ||
    compiled.lowering().launch() !=
    compiler_decode_launch_geometry(
      shape,
      maximum_context_tokens,
      workspace.split_count(),
    ) {
    raise Mismatch(Contract)
  }
  compiled
}

///|
/// Startup-only geometry planning, shared by exporter and runtime binding.
/// No source generation, autotune measurement, or model-family branching.
fn compiler_decode_launch_geometry(
  shape : @luna_attention_strategy.AttentionShape,
  context_tokens : Int,
  splits : Int,
) -> (Int, Int, Int, Int64) raise PagedAttentionAotError {
  let (query_heads, kv_heads, _, _) = shape.dimensions()
  let plan = @luna_attention_strategy.compile_attention_tile_plan(
    [],
    backend=@luna_attention_strategy.AttentionBackendId::new("cuda"),
    target=@luna_attention_strategy.AttentionTargetId::new(
      "shared-kv-decode-v2",
    ),
    problem=@luna_attention_strategy.AttentionCompileProblem::new(
      shape~,
      phase=Decode,
      numeric_type=AttentionBFloat16,
      query_rows=1,
      batch_rows=1,
      context_tokens~,
      causal=true,
    ),
    capabilities=@luna_attention_strategy.AttentionCompilerCapabilities::new(
      subgroup_width=32,
      max_threads_per_block=256,
      max_shared_memory_bytes=100000L,
      max_pipeline_stages=1,
      supports_async_copy=false,
      parallel_workgroup_target=query_heads * splits,
    ),
  ) catch {
    _ => raise Unsupported(Profile)
  }
  let candidate = plan.candidate()
  let heads = match candidate.compute() {
    AttentionGroupedQuerySplitSubgroupDot => kv_heads
    AttentionScalarDot => query_heads
    _ => raise Unsupported(Shape)
  }
  let (subgroups, _, _) = candidate.execution_geometry()
  (1, heads, subgroups * 32, candidate.shared_memory_bytes())
}

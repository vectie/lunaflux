///|
/// Exact executable ABI geometry for the blockwise F32 partition/merge law.
/// The partition count is an explicit AOT input, not inferred from latency or
/// a model name. Both source production and startup binding consume this plan.
pub fn blockwise_decode_split_launch_geometry(
  shape : @luna_attention_strategy.AttentionShape,
  splits : Int,
) -> (Int, Int, Int, Int64) raise PagedAttentionAotError {
  let (heads, kv, dimension, _) = shape.dimensions()
  let group = heads / kv
  if splits <= 1 ||
    splits > 64 ||
    group <= 0 ||
    group > 8 ||
    dimension <= 0 ||
    dimension > 256 {
    raise Unsupported(Shape)
  }
  (1, kv, group * 32, (32 * dimension * 8 + group * 32 * 4 + 16).to_int64())
}

///|
/// Offline adapter only. It specializes a legal compiler candidate through
/// semantic → ownership/effects → partition map/merge → device lowering.
/// Runtime preparation consumes only the geometry and module, never this API.
pub fn compile_blockwise_decode_split(
  shape : @luna_attention_strategy.AttentionShape,
  profile : @launch_contract.PagedKernelProfile,
  target : @catalog.DeviceTarget,
  page_stride_bytes : Int64,
  physical_page_capacity : Int64,
  maximum_context_tokens : Int,
  splits : Int,
  maximum_workspace_bytes : Int64,
  candidate_id? : Int = 452,
) -> @luna_cuda_attention_tile_aot.CudaPartitionedAttentionTileAotCompilation raise PagedAttentionAotError {
  guard candidate_id == 452 || candidate_id == 454 || candidate_id == 456 else {
    raise Unsupported(Profile)
  }
  let numeric_law = if candidate_id == 456 {
    "blockwise-fma-f32-probability-v3"
  } else if candidate_id == 454 {
    "dual-score-blockwise-f32-probability-v2"
  } else {
    "blockwise-f32-probability-v1"
  }
  let (heads, kv, dimension, page) = shape.dimensions()
  let geometry = blockwise_decode_split_launch_geometry(shape, splits)
  let compiled = @luna_cuda_attention_tile_aot.specialize_cuda_partitioned_attention_tile_aot(
    @luna_cuda_attention_tile_aot.CudaPartitionedAttentionTileAotRequest::new(
      backend=@luna_attention_strategy.AttentionBackendId::new("cuda"),
      target_id=@luna_attention_strategy.AttentionTargetId::new(numeric_law),
      problem=@luna_attention_strategy.AttentionCompileProblem::new(
        shape~,
        phase=Decode,
        numeric_type=AttentionBFloat16,
        query_rows=1,
        batch_rows=1,
        context_tokens=maximum_context_tokens,
        causal=true,
        allow_alternative_softmax_reduction=true,
      ),
      capabilities=@luna_attention_strategy.AttentionCompilerCapabilities::new(
        subgroup_width=32,
        max_threads_per_block=256,
        max_shared_memory_bytes=geometry.3,
        max_pipeline_stages=2,
        supports_async_copy=true,
        parallel_workgroup_target=kv * splits,
      ),
      device_target=target,
      maximum_shared_memory_bytes=geometry.3,
      abi=@luna_cuda_attention_tile_source.CudaPagedAttentionTileAbi::new(
        query_heads=heads,
        key_value_heads=kv,
        head_dimension=dimension,
        tokens_per_page=page,
        input_row_width=(heads + 2 * kv) * dimension,
        page_stride_values=page_stride_bytes / 2L,
        physical_page_capacity~,
        maximum_query_rows=profile.max_query_rows(),
        maximum_query_tokens=profile.max_query_tokens(),
        maximum_page_entries=profile.max_page_table_entries(),
      ),
      partial_symbol="lunaflux_paged_attention_bf16_decode_split_partial_v3_ep_3903",
      merge_symbol="lunaflux_paged_attention_bf16_decode_split_merge_v3_ep_3904",
      maximum_workspace_bytes~,
      maximum_source_bytes=1048576,
    ),
    candidate_id,
  ) catch {
    _ => raise Unsupported(Profile)
  }
  if compiled.lowering().key_value_partitions() != splits ||
    compiled.lowering().launch() != geometry {
    raise Mismatch(Contract)
  }
  compiled
}

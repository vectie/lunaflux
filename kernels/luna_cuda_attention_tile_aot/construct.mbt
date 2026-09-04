///|
pub fn CudaAttentionTileAotRequest::new(
  backend~ : @luna_attention_strategy.AttentionBackendId,
  target_id~ : @luna_attention_strategy.AttentionTargetId,
  problem~ : @luna_attention_strategy.AttentionCompileProblem,
  capabilities~ : @luna_attention_strategy.AttentionCompilerCapabilities,
  device_target~ : @catalog.DeviceTarget,
  maximum_shared_memory_bytes~ : Int64,
  abi~ : @luna_cuda_attention_tile_source.CudaPagedAttentionTileAbi,
  symbol~ : String,
  maximum_source_bytes~ : Int,
) -> CudaAttentionTileAotRequest raise CudaAttentionTileAotError {
  if maximum_shared_memory_bytes <= 0L ||
    maximum_source_bytes <= 0 ||
    maximum_source_bytes > 16777216 {
    raise InvalidCudaAttentionTileAot(InvalidRequest)
  }
  {
    backend,
    target_id,
    problem,
    capabilities,
    device_target,
    maximum_shared_memory_bytes,
    abi,
    symbol,
    maximum_source_bytes,
  }
}

///|
pub fn CudaPartitionedAttentionTileAotRequest::new(
  backend~ : @luna_attention_strategy.AttentionBackendId,
  target_id~ : @luna_attention_strategy.AttentionTargetId,
  problem~ : @luna_attention_strategy.AttentionCompileProblem,
  capabilities~ : @luna_attention_strategy.AttentionCompilerCapabilities,
  device_target~ : @catalog.DeviceTarget,
  maximum_shared_memory_bytes~ : Int64,
  abi~ : @luna_cuda_attention_tile_source.CudaPagedAttentionTileAbi,
  partial_symbol~ : String,
  merge_symbol~ : String,
  maximum_workspace_bytes~ : Int64,
  maximum_source_bytes~ : Int,
) -> CudaPartitionedAttentionTileAotRequest raise CudaAttentionTileAotError {
  if maximum_shared_memory_bytes <= 0L ||
    maximum_workspace_bytes <= 0L ||
    maximum_source_bytes <= 0 ||
    maximum_source_bytes > 16777216 ||
    partial_symbol == merge_symbol {
    raise InvalidCudaAttentionTileAot(InvalidRequest)
  }
  {
    backend,
    target_id,
    problem,
    capabilities,
    device_target,
    maximum_shared_memory_bytes,
    abi,
    partial_symbol,
    merge_symbol,
    maximum_workspace_bytes,
    maximum_source_bytes,
  }
}

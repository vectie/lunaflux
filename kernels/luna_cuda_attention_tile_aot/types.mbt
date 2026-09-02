///|
/// Explicit immutable input to the CUDA AOT compiler. Device discovery and
/// file/toolchain effects remain outside this value.
pub struct CudaAttentionTileAotRequest {
  priv backend : @luna_attention_strategy.AttentionBackendId
  priv target_id : @luna_attention_strategy.AttentionTargetId
  priv problem : @luna_attention_strategy.AttentionCompileProblem
  priv capabilities : @luna_attention_strategy.AttentionCompilerCapabilities
  priv device_target : @catalog.DeviceTarget
  priv maximum_shared_memory_bytes : Int64
  priv abi : @luna_cuda_attention_tile_source.CudaPagedAttentionTileAbi
  priv symbol : String
  priv maximum_source_bytes : Int
} derive(Debug, Eq)

///|
pub struct CudaAttentionTileAotDigest(String) derive(Debug, Eq, Hash)

///|
pub struct CudaAttentionTileAotCompilation {
  priv functional : @luna_attention_tile_compiler.ScheduledAttentionTileCompilation
  priv lowering : @luna_cuda_attention_tile_lowering.CudaAttentionTileLowering
  priv source : @luna_cuda_attention_tile_source.CudaAttentionTileSource
  priv canonical : Bytes
  priv digest : CudaAttentionTileAotDigest
} derive(Debug, Eq)

///|
pub(all) enum CudaAttentionTileAotRule {
  InvalidRequest
  FunctionalCompilation
  DeviceLowering
  SourceEmission
} derive(Debug, Eq)

///|
pub(all) suberror CudaAttentionTileAotError {
  InvalidCudaAttentionTileAot(CudaAttentionTileAotRule)
} derive(Debug, Eq)

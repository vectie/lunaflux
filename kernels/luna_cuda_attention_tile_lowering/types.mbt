///|
pub(all) enum CudaAttentionComputePrimitive {
  CudaScalarFma
  CudaSubgroupShuffleDot
  CudaGroupedQuerySubgroupShuffleDot
  CudaGroupedQuerySplitSubgroupShuffleDot
  CudaMatrixMultiplyAccumulate
  CudaMatrixQkPvMultiplyAccumulate
} derive(Debug, Eq)

///|
pub(all) enum CudaAttentionCopyPrimitive {
  CudaDirectGlobalLoad
  CudaCooperativeSharedLoad
  CudaAsynchronousSharedLoad
} derive(Debug, Eq)

///|
pub struct CudaAttentionTileLoweringDigest(String) derive(Debug, Eq, Hash)

///|
/// Concrete NVIDIA mapping. No value of this type may cross back into the
/// model, scheduler, generic attention strategy, or LunaTile semantic layers.
pub struct CudaAttentionTileLowering {
  priv program_digest : @luna_attention_tile_ir.AttentionTileProgramDigest
  priv optimization_digest : @luna_attention_tile_optimizer.AttentionTileOptimizationDigest
  priv schedule_digest : @luna_attention_tile_schedule.AttentionTileScheduleDigest
  priv target : @catalog.DeviceTarget
  priv phase : @luna_attention_strategy.AttentionExecutionPhase
  priv compute : CudaAttentionComputePrimitive
  priv copy : CudaAttentionCopyPrimitive
  priv grid_x : Int
  priv grid_y : Int
  priv key_value_partitions : Int
  priv block_threads : Int
  priv query_tile : Int
  priv key_value_tile : Int
  priv head_dimension : Int
  priv pipeline_stages : Int
  priv vector_bytes : Int
  priv shared_memory_bytes : Int64
  priv hoist_paged_key_value_page_lookup : Bool
  priv reuse_online_softmax_tile_storage : Bool
  priv matrix_storage_layout : @luna_attention_tile_schedule.AttentionTileStorageLayout?
  priv canonical : Bytes
  priv digest : CudaAttentionTileLoweringDigest
} derive(Debug, Eq)

///|
pub(all) enum CudaAttentionTileLoweringRule {
  UnsupportedTarget
  UnsupportedCompute
  UnsupportedCopy
  InvalidOptimization
  InvalidLaunch
  SharedMemory
} derive(Debug, Eq)

///|
pub(all) suberror CudaAttentionTileLoweringError {
  InvalidCudaAttentionTileLowering(CudaAttentionTileLoweringRule)
} derive(Debug, Eq)

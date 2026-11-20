///|
/// Emits CUDA only after the generic strategy, LunaTile IR, functional
/// rewrites, and immutable schedule have selected a phase-specific lowering.
pub fn emit_cuda_attention_tile_source(
  lowering : @luna_cuda_attention_tile_lowering.CudaAttentionTileLowering,
  abi : CudaPagedAttentionTileAbi,
  symbol : String,
  maximum_source_bytes : Int32,
) -> CudaAttentionTileSource raise CudaAttentionTileSourceError {
  if !valid_cuda_symbol(symbol) {
    raise InvalidCudaAttentionTileSource(InvalidSymbol)
  }
  if maximum_source_bytes <= 0 || maximum_source_bytes > 16777216 {
    raise InvalidCudaAttentionTileSource(SourceLimit)
  }
  if lowering.key_value_partitions() != 1 {
    raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
  cuda_emit_physical_attention(lowering, abi, symbol, maximum_source_bytes)
}

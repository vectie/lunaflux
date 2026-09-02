///|
pub fn CudaAttentionTileAotCompilation::functional(
  self : CudaAttentionTileAotCompilation,
) -> @luna_attention_tile_compiler.ScheduledAttentionTileCompilation {
  self.functional
}

///|
pub fn CudaAttentionTileAotCompilation::lowering(
  self : CudaAttentionTileAotCompilation,
) -> @luna_cuda_attention_tile_lowering.CudaAttentionTileLowering {
  self.lowering
}

///|
pub fn CudaAttentionTileAotCompilation::source(
  self : CudaAttentionTileAotCompilation,
) -> @luna_cuda_attention_tile_source.CudaAttentionTileSource {
  self.source
}

///|
pub fn CudaAttentionTileAotCompilation::canonical_bytes(
  self : CudaAttentionTileAotCompilation,
) -> Bytes {
  self.canonical
}

///|
pub fn CudaAttentionTileAotCompilation::digest(
  self : CudaAttentionTileAotCompilation,
) -> CudaAttentionTileAotDigest {
  self.digest
}

///|
pub fn CudaAttentionTileAotDigest::as_hex(
  self : CudaAttentionTileAotDigest,
) -> String {
  self.0
}

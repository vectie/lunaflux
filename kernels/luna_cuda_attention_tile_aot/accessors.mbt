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
pub fn CudaAttentionTileAotFrontier::functional(
  self : CudaAttentionTileAotFrontier,
) -> @luna_attention_tile_compiler.AttentionTileCompilationFrontier {
  self.functional
}

///|
pub fn CudaAttentionTileAotFrontier::selected(
  self : CudaAttentionTileAotFrontier,
) -> CudaAttentionTileAotCompilation {
  self.selected
}

///|
pub fn CudaAttentionTileAotFrontier::variants(
  self : CudaAttentionTileAotFrontier,
) -> ReadOnlyArray[CudaAttentionTileAotCompilation] {
  self.variants
}

///|
pub fn CudaAttentionTileAotFrontier::canonical_bytes(
  self : CudaAttentionTileAotFrontier,
) -> Bytes {
  self.canonical
}

///|
pub fn CudaAttentionTileAotFrontier::digest(
  self : CudaAttentionTileAotFrontier,
) -> CudaAttentionTileAotDigest {
  self.digest
}

///|
pub fn CudaAttentionTileAotDigest::as_hex(
  self : CudaAttentionTileAotDigest,
) -> String {
  self.0
}

///|
pub fn CudaPartitionedAttentionTileAotCompilation::functional(
  self : CudaPartitionedAttentionTileAotCompilation,
) -> @luna_attention_tile_compiler.ScheduledAttentionTileCompilation {
  self.functional
}

///|
pub fn CudaPartitionedAttentionTileAotCompilation::lowering(
  self : CudaPartitionedAttentionTileAotCompilation,
) -> @luna_cuda_attention_tile_lowering.CudaAttentionTileLowering {
  self.lowering
}

///|
pub fn CudaPartitionedAttentionTileAotCompilation::source(
  self : CudaPartitionedAttentionTileAotCompilation,
) -> @luna_cuda_attention_tile_source.CudaPartitionedAttentionTileSource {
  self.source
}

///|
pub fn CudaPartitionedAttentionTileAotCompilation::canonical_bytes(
  self : CudaPartitionedAttentionTileAotCompilation,
) -> Bytes {
  self.canonical
}

///|
pub fn CudaPartitionedAttentionTileAotCompilation::digest(
  self : CudaPartitionedAttentionTileAotCompilation,
) -> CudaAttentionTileAotDigest {
  self.digest
}

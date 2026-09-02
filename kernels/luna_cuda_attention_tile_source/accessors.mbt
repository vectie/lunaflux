///|
pub fn CudaPagedAttentionTileAbi::shape(
  self : CudaPagedAttentionTileAbi,
) -> (Int, Int, Int, Int) {
  (
    self.query_heads,
    self.key_value_heads,
    self.head_dimension,
    self.tokens_per_page,
  )
}

///|
pub fn CudaPagedAttentionTileAbi::layout(
  self : CudaPagedAttentionTileAbi,
) -> (Int, Int64, Int64) {
  (self.input_row_width, self.page_stride_values, self.physical_page_capacity)
}

///|
pub fn CudaPagedAttentionTileAbi::limits(
  self : CudaPagedAttentionTileAbi,
) -> (Int, Int, Int) {
  (
    self.maximum_query_rows,
    self.maximum_query_tokens,
    self.maximum_page_entries,
  )
}

///|
pub fn CudaAttentionTileSource::lowering_digest(
  self : CudaAttentionTileSource,
) -> @luna_cuda_attention_tile_lowering.CudaAttentionTileLoweringDigest {
  self.lowering_digest
}

///|
pub fn CudaAttentionTileSource::abi(
  self : CudaAttentionTileSource,
) -> CudaPagedAttentionTileAbi {
  self.abi
}

///|
pub fn CudaAttentionTileSource::symbol(
  self : CudaAttentionTileSource,
) -> StringView {
  self.symbol
}

///|
pub fn CudaAttentionTileSource::bytes(self : CudaAttentionTileSource) -> Bytes {
  self.bytes
}

///|
pub fn CudaAttentionTileSource::digest(
  self : CudaAttentionTileSource,
) -> CudaAttentionTileSourceDigest {
  self.digest
}

///|
pub fn CudaAttentionTileSourceDigest::as_hex(
  self : CudaAttentionTileSourceDigest,
) -> String {
  self.0
}

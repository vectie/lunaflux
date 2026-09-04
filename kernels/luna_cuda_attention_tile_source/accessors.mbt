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

///|
pub fn CudaPartitionedAttentionWorkspace::region_offsets_bytes(
  self : CudaPartitionedAttentionWorkspace,
) -> (Int64, Int64, Int64) {
  (
    self.maximum_offset_bytes,
    self.denominator_offset_bytes,
    self.numerator_offset_bytes,
  )
}

///|
pub fn CudaPartitionedAttentionWorkspace::partial_state_count(
  self : CudaPartitionedAttentionWorkspace,
) -> Int64 {
  self.partial_state_count
}

///|
pub fn CudaPartitionedAttentionWorkspace::total_bytes(
  self : CudaPartitionedAttentionWorkspace,
) -> Int64 {
  self.total_bytes
}

///|
pub fn CudaPartitionedAttentionTileSource::lowering_digest(
  self : CudaPartitionedAttentionTileSource,
) -> @luna_cuda_attention_tile_lowering.CudaAttentionTileLoweringDigest {
  self.lowering_digest
}

///|
pub fn CudaPartitionedAttentionTileSource::abi(
  self : CudaPartitionedAttentionTileSource,
) -> CudaPagedAttentionTileAbi {
  self.abi
}

///|
pub fn CudaPartitionedAttentionTileSource::symbols(
  self : CudaPartitionedAttentionTileSource,
) -> (StringView, StringView) {
  (self.partial_symbol, self.merge_symbol)
}

///|
pub fn CudaPartitionedAttentionTileSource::workspace(
  self : CudaPartitionedAttentionTileSource,
) -> CudaPartitionedAttentionWorkspace {
  self.workspace
}

///|
pub fn CudaPartitionedAttentionTileSource::bytes(
  self : CudaPartitionedAttentionTileSource,
) -> Bytes {
  self.bytes
}

///|
pub fn CudaPartitionedAttentionTileSource::digest(
  self : CudaPartitionedAttentionTileSource,
) -> CudaAttentionTileSourceDigest {
  self.digest
}

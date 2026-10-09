///|
pub fn DecoderSelectedSparseAttentionCudaSource::operation(
  self : Self,
) -> DecoderSelectedSparseAttentionCudaOperation {
  self.operation
}

///|
pub fn DecoderSelectedSparseAttentionCudaSource::numerical_contract(
  self : Self,
) -> DecoderSelectedSparseAttentionNumericalContract {
  self.numerical_contract
}

///|
pub fn DecoderSelectedSparseAttentionCudaSource::query_head_count(
  self : Self,
) -> Int {
  self.query_head_count
}

///|
pub fn DecoderSelectedSparseAttentionCudaSource::head_dimension(
  self : Self,
) -> Int {
  self.head_dimension
}

///|
pub fn DecoderSelectedSparseAttentionCudaSource::selected_width(
  self : Self,
) -> Int {
  self.selected_width
}

///|
pub fn DecoderSelectedSparseAttentionCudaSource::maximum_key_slots(
  self : Self,
) -> Int {
  self.maximum_key_slots
}

///|
pub fn DecoderSelectedSparseAttentionCudaSource::maximum_rows(
  self : Self,
) -> Int {
  self.maximum_rows
}

///|
pub fn DecoderSelectedSparseAttentionCudaSource::maximum_batches(
  self : Self,
) -> Int {
  self.maximum_batches
}

///|
pub fn DecoderSelectedSparseAttentionCudaSource::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn DecoderSelectedSparseAttentionCudaSource::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

///|
pub fn GlmHybridSparseAttentionCudaSource::operation(
  self : Self,
) -> GlmHybridSparseAttentionCudaOperation {
  self.operation
}

///|
pub fn GlmHybridSparseAttentionCudaSource::numerical_contract(
  self : Self,
) -> GlmHybridSparseAttentionNumericalContract {
  self.numerical_contract
}

///|
pub fn GlmHybridSparseAttentionCudaSource::query_head_count(self : Self) -> Int {
  self.query_head_count
}

///|
pub fn GlmHybridSparseAttentionCudaSource::key_value_head_count(
  self : Self,
) -> Int {
  self.key_value_head_count
}

///|
pub fn GlmHybridSparseAttentionCudaSource::query_key_dimension(
  self : Self,
) -> Int {
  self.query_key_dimension
}

///|
pub fn GlmHybridSparseAttentionCudaSource::value_dimension(self : Self) -> Int {
  self.value_dimension
}

///|
pub fn GlmHybridSparseAttentionCudaSource::selected_width(self : Self) -> Int {
  self.selected_width
}

///|
pub fn GlmHybridSparseAttentionCudaSource::maximum_tokens(self : Self) -> Int {
  self.maximum_tokens
}

///|
pub fn GlmHybridSparseAttentionCudaSource::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn GlmHybridSparseAttentionCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

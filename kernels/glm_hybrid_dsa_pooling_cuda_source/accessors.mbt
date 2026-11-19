///|
pub fn GlmHybridDsaPoolingCudaSource::operation(
  self : Self,
) -> GlmHybridDsaPoolingCudaOperation {
  self.operation
}

///|
pub fn GlmHybridDsaPoolingCudaSource::numerical_contract(
  self : Self,
) -> GlmHybridDsaPoolingNumericalContract {
  self.numerical_contract
}

///|
pub fn GlmHybridDsaPoolingCudaSource::head_dimension(self : Self) -> Int32 {
  self.head_dimension
}

///|
pub fn GlmHybridDsaPoolingCudaSource::pool_size(self : Self) -> Int32 {
  self.pool_size
}

///|
pub fn GlmHybridDsaPoolingCudaSource::maximum_tokens(self : Self) -> Int32 {
  self.maximum_tokens
}

///|
pub fn GlmHybridDsaPoolingCudaSource::pool_capacity(self : Self) -> Int32 {
  self.pool_capacity
}

///|
pub fn GlmHybridDsaPoolingCudaSource::function_symbol(self : Self) -> String {
  self.function_symbol
}

///|
pub fn GlmHybridDsaPoolingCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

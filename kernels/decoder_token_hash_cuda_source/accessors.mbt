///|
pub fn DecoderTokenHashCudaSource::operation(
  self : Self,
) -> DecoderTokenHashCudaOperation {
  self.operation
}

///|
pub fn DecoderTokenHashCudaSource::numerical_contract(
  self : Self,
) -> DecoderTokenHashNumericalContract {
  self.numerical_contract
}

///|
pub fn DecoderTokenHashCudaSource::vocabulary_size(self : Self) -> Int {
  self.vocabulary_size
}

///|
pub fn DecoderTokenHashCudaSource::experts_per_token(self : Self) -> Int {
  self.experts_per_token
}

///|
pub fn DecoderTokenHashCudaSource::routed_expert_count(self : Self) -> Int {
  self.routed_expert_count
}

///|
pub fn DecoderTokenHashCudaSource::maximum_rows(self : Self) -> Int {
  self.maximum_rows
}

///|
pub fn DecoderTokenHashCudaSource::function_symbol(self : Self) -> String {
  self.function_symbol
}

///|
pub fn DecoderTokenHashCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

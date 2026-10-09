///|
pub fn DecoderRoutingCudaSource::operation(
  self : Self,
) -> DecoderRoutingCudaOperation {
  self.operation
}

///|
pub fn DecoderRoutingCudaSource::numerical_contract(
  self : Self,
) -> DecoderRoutingNumericalContract {
  self.numerical_contract
}

///|
pub fn DecoderRoutingCudaSource::routed_expert_count(self : Self) -> Int {
  self.routed_expert_count
}

///|
pub fn DecoderRoutingCudaSource::experts_per_token(self : Self) -> Int {
  self.experts_per_token
}

///|
pub fn DecoderRoutingCudaSource::maximum_rows(self : Self) -> Int {
  self.maximum_rows
}

///|
pub fn DecoderRoutingCudaSource::function_symbol(self : Self) -> String {
  self.function_symbol
}

///|
pub fn DecoderRoutingCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

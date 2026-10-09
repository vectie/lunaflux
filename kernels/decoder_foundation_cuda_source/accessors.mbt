///|
pub fn DecoderFoundationCudaSource::operation(
  self : Self,
) -> DecoderFoundationCudaOperation {
  self.operation
}

///|
pub fn DecoderFoundationCudaSource::numerical_contract(
  self : Self,
) -> DecoderFoundationNumericalContract {
  self.numerical_contract
}

///|
pub fn DecoderFoundationCudaSource::function_symbol(self : Self) -> String {
  self.function_symbol
}

///|
pub fn DecoderFoundationCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

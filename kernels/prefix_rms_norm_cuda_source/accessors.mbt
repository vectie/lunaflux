///|
pub fn PrefixRmsNormCudaSource::operation(
  self : Self,
) -> PrefixRmsNormCudaOperation {
  self.operation
}

///|
pub fn PrefixRmsNormCudaSource::numerical_contract(
  self : Self,
) -> PrefixRmsNormNumericalContract {
  self.numerical_contract
}

///|
pub fn PrefixRmsNormCudaSource::prefix_width(self : Self) -> Int {
  self.prefix_width
}

///|
pub fn PrefixRmsNormCudaSource::suffix_width(self : Self) -> Int {
  self.suffix_width
}

///|
pub fn PrefixRmsNormCudaSource::maximum_rows(self : Self) -> Int {
  self.maximum_rows
}

///|
pub fn PrefixRmsNormCudaSource::epsilon(self : Self) -> Double {
  self.epsilon
}

///|
pub fn PrefixRmsNormCudaSource::function_symbol(self : Self) -> String {
  self.function_symbol
}

///|
pub fn PrefixRmsNormCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

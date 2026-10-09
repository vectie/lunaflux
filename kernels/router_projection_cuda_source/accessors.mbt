///|
pub fn RouterProjectionCudaSource::operation(
  self : Self,
) -> RouterProjectionCudaOperation {
  self.operation
}

///|
pub fn RouterProjectionCudaSource::numerical_contract(
  self : Self,
) -> RouterProjectionNumericalContract {
  self.numerical_contract
}

///|
pub fn RouterProjectionCudaSource::input_width(self : Self) -> Int {
  self.input_width
}

///|
pub fn RouterProjectionCudaSource::output_width(self : Self) -> Int {
  self.output_width
}

///|
pub fn RouterProjectionCudaSource::maximum_rows(self : Self) -> Int {
  self.maximum_rows
}

///|
pub fn RouterProjectionCudaSource::function_symbol(self : Self) -> String {
  self.function_symbol
}

///|
pub fn RouterProjectionCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

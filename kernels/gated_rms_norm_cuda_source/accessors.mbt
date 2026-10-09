///|
pub fn GatedRmsNormCudaSource::operation(
  self : Self,
) -> GatedRmsNormCudaOperation {
  self.operation
}

///|
pub fn GatedRmsNormCudaSource::numerical_contract(
  self : Self,
) -> GatedRmsNormNumericalContract {
  self.numerical_contract
}

///|
pub fn GatedRmsNormCudaSource::head_count(self : Self) -> Int {
  self.head_count
}

///|
pub fn GatedRmsNormCudaSource::head_dimension(self : Self) -> Int {
  self.head_dimension
}

///|
pub fn GatedRmsNormCudaSource::maximum_rows(self : Self) -> Int {
  self.maximum_rows
}

///|
pub fn GatedRmsNormCudaSource::epsilon(self : Self) -> Double {
  self.epsilon
}

///|
pub fn GatedRmsNormCudaSource::function_symbol(self : Self) -> String {
  self.function_symbol
}

///|
pub fn GatedRmsNormCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

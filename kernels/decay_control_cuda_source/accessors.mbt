///|
pub fn DecayControlCudaSource::operation(
  self : Self,
) -> DecayControlCudaOperation {
  self.operation
}

///|
pub fn DecayControlCudaSource::numerical_contract(
  self : Self,
) -> DecayControlNumericalContract {
  self.numerical_contract
}

///|
pub fn DecayControlCudaSource::head_count(self : Self) -> Int32 {
  self.head_count
}

///|
pub fn DecayControlCudaSource::head_dimension(self : Self) -> Int32 {
  self.head_dimension
}

///|
pub fn DecayControlCudaSource::lower_bound(self : Self) -> Double {
  self.lower_bound
}

///|
pub fn DecayControlCudaSource::maximum_rows(self : Self) -> Int32 {
  self.maximum_rows
}

///|
pub fn DecayControlCudaSource::function_symbol(self : Self) -> String {
  self.function_symbol
}

///|
pub fn DecayControlCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

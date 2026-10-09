///|
pub fn AffineModulationCudaSource::operation(
  self : Self,
) -> AffineModulationCudaOperation {
  self.operation
}

///|
pub fn AffineModulationCudaSource::numerical_contract(
  self : Self,
) -> AffineModulationNumericalContract {
  self.numerical_contract
}

///|
pub fn AffineModulationCudaSource::row_count(self : Self) -> Int64 {
  self.row_count
}

///|
pub fn AffineModulationCudaSource::width(self : Self) -> Int {
  self.width
}

///|
pub fn AffineModulationCudaSource::function_symbol(self : Self) -> String {
  self.function_symbol
}

///|
pub fn AffineModulationCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn SigmoidCorrectionCudaSource::operation(
  self : Self,
) -> SigmoidCorrectionCudaOperation {
  self.operation
}

///|
pub fn SigmoidCorrectionCudaSource::numerical_contract(
  self : Self,
) -> SigmoidCorrectionNumericalContract {
  self.numerical_contract
}

///|
pub fn SigmoidCorrectionCudaSource::expert_count(self : Self) -> Int {
  self.expert_count
}

///|
pub fn SigmoidCorrectionCudaSource::maximum_rows(self : Self) -> Int {
  self.maximum_rows
}

///|
pub fn SigmoidCorrectionCudaSource::function_symbol(self : Self) -> String {
  self.function_symbol
}

///|
pub fn SigmoidCorrectionCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

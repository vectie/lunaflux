///|
pub fn SelectedExpertMlpCudaSource::numerical_contract(
  self : Self,
) -> SelectedExpertMlpNumericalContract {
  self.numerical_contract
}

///|
pub fn SelectedExpertMlpCudaSource::activation(
  self : Self,
) -> SelectedExpertActivation {
  self.activation
}

///|
pub fn SelectedExpertMlpCudaSource::hidden_size(self : Self) -> Int {
  self.hidden_size
}

///|
pub fn SelectedExpertMlpCudaSource::intermediate_size(self : Self) -> Int {
  self.intermediate_size
}

///|
pub fn SelectedExpertMlpCudaSource::routed_expert_count(self : Self) -> Int {
  self.routed_expert_count
}

///|
pub fn SelectedExpertMlpCudaSource::experts_per_token(self : Self) -> Int {
  self.experts_per_token
}

///|
pub fn SelectedExpertMlpCudaSource::maximum_rows(self : Self) -> Int {
  self.maximum_rows
}

///|
pub fn SelectedExpertMlpCudaSource::function_symbol(self : Self) -> String {
  self.function_symbol
}

///|
pub fn SelectedExpertMlpCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

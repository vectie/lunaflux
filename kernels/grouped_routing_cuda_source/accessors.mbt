///|
pub fn GroupedRoutingCudaSource::operation(
  self : Self,
) -> GroupedRoutingCudaOperation {
  self.operation
}

///|
pub fn GroupedRoutingCudaSource::numerical_contract(
  self : Self,
) -> GroupedRoutingNumericalContract {
  self.numerical_contract
}

///|
pub fn GroupedRoutingCudaSource::routed_expert_count(self : Self) -> Int {
  self.routed_expert_count
}

///|
pub fn GroupedRoutingCudaSource::group_count(self : Self) -> Int {
  self.group_count
}

///|
pub fn GroupedRoutingCudaSource::group_score_topk(self : Self) -> Int {
  self.group_score_topk
}

///|
pub fn GroupedRoutingCudaSource::selected_group_count(self : Self) -> Int {
  self.selected_group_count
}

///|
pub fn GroupedRoutingCudaSource::experts_per_token(self : Self) -> Int {
  self.experts_per_token
}

///|
pub fn GroupedRoutingCudaSource::normalize(self : Self) -> Bool {
  self.normalize
}

///|
pub fn GroupedRoutingCudaSource::scaling(self : Self) -> Double {
  self.scaling
}

///|
pub fn GroupedRoutingCudaSource::maximum_rows(self : Self) -> Int {
  self.maximum_rows
}

///|
pub fn GroupedRoutingCudaSource::function_symbol(self : Self) -> String {
  self.function_symbol
}

///|
pub fn GroupedRoutingCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

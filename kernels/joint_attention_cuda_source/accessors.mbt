///|
pub fn JointAttentionCudaSource::numerical_contract(
  self : Self,
) -> JointAttentionNumericalContract {
  self.numerical_contract
}

///|
pub fn JointAttentionCudaSource::row_count(self : Self) -> Int {
  self.row_count
}

///|
pub fn JointAttentionCudaSource::hidden_size(self : Self) -> Int {
  self.hidden_size
}

///|
pub fn JointAttentionCudaSource::head_count(self : Self) -> Int {
  self.head_count
}

///|
pub fn JointAttentionCudaSource::head_dimension(self : Self) -> Int {
  self.head_dimension
}

///|
pub fn JointAttentionCudaSource::rotary_dimension(self : Self) -> Int {
  self.rotary_dimension
}

///|
pub fn JointAttentionCudaSource::applies_rotary(self : Self) -> Bool {
  self.applies_rotary
}

///|
pub fn JointAttentionCudaSource::function_prefix(self : Self) -> String {
  self.function_prefix
}

///|
pub fn JointAttentionCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

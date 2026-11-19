///|
pub fn JointAttentionCudaSource::numerical_contract(
  self : Self,
) -> JointAttentionNumericalContract {
  self.numerical_contract
}

///|
pub fn JointAttentionCudaSource::row_count(self : Self) -> Int32 {
  self.row_count
}

///|
pub fn JointAttentionCudaSource::hidden_size(self : Self) -> Int32 {
  self.hidden_size
}

///|
pub fn JointAttentionCudaSource::head_count(self : Self) -> Int32 {
  self.head_count
}

///|
pub fn JointAttentionCudaSource::head_dimension(self : Self) -> Int32 {
  self.head_dimension
}

///|
pub fn JointAttentionCudaSource::rotary_dimension(self : Self) -> Int32 {
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

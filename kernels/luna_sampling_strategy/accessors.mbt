///|
pub fn SamplingDevicePlan::maximum_rows(self : SamplingDevicePlan) -> Int {
  self.maximum_rows
}

///|
pub fn SamplingDevicePlan::vocabulary_size(self : SamplingDevicePlan) -> Int {
  self.vocabulary_size
}

///|
pub fn SamplingDevicePlan::maximum_top_k(self : SamplingDevicePlan) -> Int {
  self.maximum_top_k
}

///|
pub fn SamplingDevicePlan::parameter_row_bytes(
  self : SamplingDevicePlan,
) -> Int64 {
  self.parameter_row_bytes
}

///|
pub fn SamplingDevicePlan::result_row_bytes(self : SamplingDevicePlan) -> Int64 {
  self.result_row_bytes
}

///|
pub fn SamplingDevicePlan::candidate_token_offset_bytes(
  self : SamplingDevicePlan,
) -> Int64 {
  self.candidate_token_offset_bytes
}

///|
pub fn SamplingDevicePlan::candidate_weight_offset_bytes(
  self : SamplingDevicePlan,
) -> Int64 {
  self.candidate_weight_offset_bytes
}

///|
pub fn SamplingDevicePlan::workspace_bytes(self : SamplingDevicePlan) -> Int64 {
  self.workspace_bytes
}

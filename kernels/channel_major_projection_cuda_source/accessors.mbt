///|
pub fn ChannelMajorProjectionCudaSource::operation(
  self : Self,
) -> ChannelMajorProjectionOperation {
  self.operation
}

///|
pub fn ChannelMajorProjectionCudaSource::numerical_contract(
  self : Self,
) -> ChannelMajorProjectionNumericalContract {
  self.numerical_contract
}

///|
pub fn ChannelMajorProjectionCudaSource::batch_count(self : Self) -> Int {
  self.batch_count
}

///|
pub fn ChannelMajorProjectionCudaSource::input_channels(self : Self) -> Int {
  self.input_channels
}

///|
pub fn ChannelMajorProjectionCudaSource::output_channels(self : Self) -> Int {
  self.output_channels
}

///|
pub fn ChannelMajorProjectionCudaSource::time_count(self : Self) -> Int {
  self.time_count
}

///|
pub fn ChannelMajorProjectionCudaSource::input_element_count(
  self : Self,
) -> Int64 {
  self.input_elements
}

///|
pub fn ChannelMajorProjectionCudaSource::output_element_count(
  self : Self,
) -> Int64 {
  self.output_elements
}

///|
pub fn ChannelMajorProjectionCudaSource::function_symbol(self : Self) -> String {
  self.function_symbol
}

///|
pub fn ChannelMajorProjectionCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

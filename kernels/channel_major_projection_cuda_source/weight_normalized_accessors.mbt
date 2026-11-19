///|
pub fn WeightNormalizedConv1dCudaSource::operation(
  self : Self,
) -> ChannelMajorProjectionOperation {
  self.operation
}

///|
pub fn WeightNormalizedConv1dCudaSource::numerical_contract(
  self : Self,
) -> ChannelMajorProjectionNumericalContract {
  self.numerical_contract
}

///|
pub fn WeightNormalizedConv1dCudaSource::batch_count(self : Self) -> Int32 {
  self.batch_count
}

///|
pub fn WeightNormalizedConv1dCudaSource::input_channels(self : Self) -> Int32 {
  self.input_channels
}

///|
pub fn WeightNormalizedConv1dCudaSource::output_channels(self : Self) -> Int32 {
  self.output_channels
}

///|
pub fn WeightNormalizedConv1dCudaSource::input_time(self : Self) -> Int32 {
  self.input_time
}

///|
pub fn WeightNormalizedConv1dCudaSource::output_time(self : Self) -> Int32 {
  self.output_time
}

///|
pub fn WeightNormalizedConv1dCudaSource::kernel_size(self : Self) -> Int32 {
  self.kernel_size
}

///|
pub fn WeightNormalizedConv1dCudaSource::padding(self : Self) -> Int32 {
  self.padding
}

///|
pub fn WeightNormalizedConv1dCudaSource::stride(self : Self) -> Int32 {
  self.stride
}

///|
pub fn WeightNormalizedConv1dCudaSource::input_element_count(
  self : Self,
) -> Int64 {
  self.input_elements
}

///|
pub fn WeightNormalizedConv1dCudaSource::weight_element_count(
  self : Self,
) -> Int64 {
  self.weight_elements
}

///|
pub fn WeightNormalizedConv1dCudaSource::output_element_count(
  self : Self,
) -> Int64 {
  self.output_elements
}

///|
pub fn WeightNormalizedConv1dCudaSource::normalize_symbol(
  self : Self,
) -> String {
  self.normalize_symbol
}

///|
pub fn WeightNormalizedConv1dCudaSource::convolution_symbol(
  self : Self,
) -> String {
  self.convolution_symbol
}

///|
pub fn WeightNormalizedConv1dCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn WeightNormalizedConvTranspose1dCudaSource::operation(
  self : Self,
) -> ChannelMajorProjectionOperation {
  self.operation
}

///|
pub fn WeightNormalizedConvTranspose1dCudaSource::numerical_contract(
  self : Self,
) -> ChannelMajorProjectionNumericalContract {
  self.numerical_contract
}

///|
pub fn WeightNormalizedConvTranspose1dCudaSource::batch_count(
  self : Self,
) -> Int {
  self.batch_count
}

///|
pub fn WeightNormalizedConvTranspose1dCudaSource::input_channels(
  self : Self,
) -> Int {
  self.input_channels
}

///|
pub fn WeightNormalizedConvTranspose1dCudaSource::output_channels(
  self : Self,
) -> Int {
  self.output_channels
}

///|
pub fn WeightNormalizedConvTranspose1dCudaSource::input_time(
  self : Self,
) -> Int {
  self.input_time
}

///|
pub fn WeightNormalizedConvTranspose1dCudaSource::output_time(
  self : Self,
) -> Int {
  self.output_time
}

///|
pub fn WeightNormalizedConvTranspose1dCudaSource::kernel_size(
  self : Self,
) -> Int {
  self.kernel_size
}

///|
pub fn WeightNormalizedConvTranspose1dCudaSource::padding(self : Self) -> Int {
  self.padding
}

///|
pub fn WeightNormalizedConvTranspose1dCudaSource::stride(self : Self) -> Int {
  self.stride
}

///|
pub fn WeightNormalizedConvTranspose1dCudaSource::input_element_count(
  self : Self,
) -> Int64 {
  self.input_elements
}

///|
pub fn WeightNormalizedConvTranspose1dCudaSource::weight_element_count(
  self : Self,
) -> Int64 {
  self.weight_elements
}

///|
pub fn WeightNormalizedConvTranspose1dCudaSource::output_element_count(
  self : Self,
) -> Int64 {
  self.output_elements
}

///|
pub fn WeightNormalizedConvTranspose1dCudaSource::normalize_symbol(
  self : Self,
) -> String {
  self.normalize_symbol
}

///|
pub fn WeightNormalizedConvTranspose1dCudaSource::convolution_symbol(
  self : Self,
) -> String {
  self.convolution_symbol
}

///|
pub fn WeightNormalizedConvTranspose1dCudaSource::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

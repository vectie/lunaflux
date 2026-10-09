///|
pub fn WeightNormalizedDilatedConv1dCudaSource::operation(
  self : Self,
) -> ChannelMajorProjectionOperation {
  self.operation
}

///|
pub fn WeightNormalizedDilatedConv1dCudaSource::numerical_contract(
  self : Self,
) -> ChannelMajorProjectionNumericalContract {
  self.numerical_contract
}

///|
pub fn WeightNormalizedDilatedConv1dCudaSource::batch_count(self : Self) -> Int {
  self.batch_count
}

///|
pub fn WeightNormalizedDilatedConv1dCudaSource::channel_count(
  self : Self,
) -> Int {
  self.channel_count
}

///|
pub fn WeightNormalizedDilatedConv1dCudaSource::time_count(self : Self) -> Int {
  self.time_count
}

///|
pub fn WeightNormalizedDilatedConv1dCudaSource::kernel_size(self : Self) -> Int {
  self.kernel_size
}

///|
pub fn WeightNormalizedDilatedConv1dCudaSource::dilation(self : Self) -> Int {
  self.dilation
}

///|
pub fn WeightNormalizedDilatedConv1dCudaSource::padding(self : Self) -> Int {
  self.padding
}

///|
pub fn WeightNormalizedDilatedConv1dCudaSource::element_count(
  self : Self,
) -> Int64 {
  self.element_count
}

///|
pub fn WeightNormalizedDilatedConv1dCudaSource::weight_elements(
  self : Self,
) -> Int64 {
  self.weight_elements
}

///|
pub fn WeightNormalizedDilatedConv1dCudaSource::normalize_symbol(
  self : Self,
) -> String {
  self.normalize_symbol
}

///|
pub fn WeightNormalizedDilatedConv1dCudaSource::convolution_symbol(
  self : Self,
) -> String {
  self.convolution_symbol
}

///|
pub fn WeightNormalizedDilatedConv1dCudaSource::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

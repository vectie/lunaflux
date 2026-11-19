///|
pub fn AliasFreeActivationCudaSource::numerical_contract(
  self : Self,
) -> AliasFreeActivationNumericalContract {
  self.numerical_contract
}

///|
pub fn AliasFreeActivationCudaSource::batch_count(self : Self) -> Int32 {
  self.batch_count
}

///|
pub fn AliasFreeActivationCudaSource::channel_count(self : Self) -> Int32 {
  self.channel_count
}

///|
pub fn AliasFreeActivationCudaSource::time_count(self : Self) -> Int32 {
  self.time_count
}

///|
pub fn AliasFreeActivationCudaSource::ratio(self : Self) -> Int32 {
  self.ratio
}

///|
pub fn AliasFreeActivationCudaSource::kernel_size(self : Self) -> Int32 {
  self.kernel_size
}

///|
pub fn AliasFreeActivationCudaSource::upsample_time(self : Self) -> Int32 {
  self.upsample_time
}

///|
pub fn AliasFreeActivationCudaSource::input_element_count(self : Self) -> Int64 {
  self.input_element_count
}

///|
pub fn AliasFreeActivationCudaSource::intermediate_element_count(
  self : Self,
) -> Int64 {
  self.intermediate_element_count
}

///|
pub fn AliasFreeActivationCudaSource::output_element_count(
  self : Self,
) -> Int64 {
  self.output_element_count
}

///|
pub fn AliasFreeActivationCudaSource::upsample_symbol(self : Self) -> String {
  self.upsample_symbol
}

///|
pub fn AliasFreeActivationCudaSource::activation_symbol(self : Self) -> String {
  self.activation_symbol
}

///|
pub fn AliasFreeActivationCudaSource::downsample_symbol(self : Self) -> String {
  self.downsample_symbol
}

///|
pub fn AliasFreeActivationCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

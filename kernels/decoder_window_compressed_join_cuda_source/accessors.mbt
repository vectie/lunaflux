///|
pub fn DecoderWindowCompressedJoinCudaSource::operation(
  self : Self,
) -> DecoderWindowCompressedJoinCudaOperation {
  self.operation
}

///|
pub fn DecoderWindowCompressedJoinCudaSource::numerical_contract(
  self : Self,
) -> DecoderWindowCompressedJoinNumericalContract {
  self.numerical_contract
}

///|
pub fn DecoderWindowCompressedJoinCudaSource::window_size(self : Self) -> Int {
  self.window_size
}

///|
pub fn DecoderWindowCompressedJoinCudaSource::compression_ratio(
  self : Self,
) -> Int {
  self.compression_ratio
}

///|
pub fn DecoderWindowCompressedJoinCudaSource::maximum_position_count(
  self : Self,
) -> Int {
  self.maximum_position_count
}

///|
pub fn DecoderWindowCompressedJoinCudaSource::maximum_compressed_width(
  self : Self,
) -> Int {
  self.maximum_compressed_width
}

///|
pub fn DecoderWindowCompressedJoinCudaSource::maximum_joined_width(
  self : Self,
) -> Int {
  self.maximum_joined_width
}

///|
pub fn DecoderWindowCompressedJoinCudaSource::maximum_rows(self : Self) -> Int {
  self.maximum_rows
}

///|
pub fn DecoderWindowCompressedJoinCudaSource::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn DecoderWindowCompressedJoinCudaSource::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

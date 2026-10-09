///|
pub fn DecoderCompressedIndexCudaSource::operation(
  self : Self,
) -> DecoderCompressedIndexCudaOperation {
  self.operation
}

///|
pub fn DecoderCompressedIndexCudaSource::numerical_contract(
  self : Self,
) -> DecoderCompressedIndexNumericalContract {
  self.numerical_contract
}

///|
pub fn DecoderCompressedIndexCudaSource::compression_ratio(self : Self) -> Int {
  self.compression_ratio
}

///|
pub fn DecoderCompressedIndexCudaSource::maximum_position_count(
  self : Self,
) -> Int {
  self.maximum_position_count
}

///|
pub fn DecoderCompressedIndexCudaSource::maximum_slots_per_row(
  self : Self,
) -> Int {
  self.maximum_slots_per_row
}

///|
pub fn DecoderCompressedIndexCudaSource::maximum_rows(self : Self) -> Int {
  self.maximum_rows
}

///|
pub fn DecoderCompressedIndexCudaSource::function_symbol(self : Self) -> String {
  self.function_symbol
}

///|
pub fn DecoderCompressedIndexCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn DecoderCompressedKvPrepareCudaSource::operation(
  self : Self,
) -> DecoderCompressedKvPrepareOperation {
  self.operation
}

///|
pub fn DecoderCompressedKvPrepareCudaSource::numerical_contract(
  self : Self,
) -> DecoderCompressedKvPrepareNumericalContract {
  self.numerical_contract
}

///|
pub fn DecoderCompressedKvPrepareCudaSource::window_size(self : Self) -> Int32 {
  self.window_size
}

///|
pub fn DecoderCompressedKvPrepareCudaSource::compression_ratio(
  self : Self,
) -> Int32 {
  self.compression_ratio
}

///|
pub fn DecoderCompressedKvPrepareCudaSource::head_dimension(
  self : Self,
) -> Int32 {
  self.head_dimension
}

///|
pub fn DecoderCompressedKvPrepareCudaSource::maximum_position_count(
  self : Self,
) -> Int32 {
  self.maximum_position_count
}

///|
pub fn DecoderCompressedKvPrepareCudaSource::maximum_compressed_slots(
  self : Self,
) -> Int32 {
  self.maximum_compressed_slots
}

///|
pub fn DecoderCompressedKvPrepareCudaSource::maximum_key_slots(
  self : Self,
) -> Int32 {
  self.maximum_key_slots
}

///|
pub fn DecoderCompressedKvPrepareCudaSource::maximum_batches(
  self : Self,
) -> Int32 {
  self.maximum_batches
}

///|
pub fn DecoderCompressedKvPrepareCudaSource::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn DecoderCompressedKvPrepareCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

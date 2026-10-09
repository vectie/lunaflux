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
pub fn DecoderCompressedKvPrepareCudaSource::window_size(self : Self) -> Int {
  self.window_size
}

///|
pub fn DecoderCompressedKvPrepareCudaSource::compression_ratio(
  self : Self,
) -> Int {
  self.compression_ratio
}

///|
pub fn DecoderCompressedKvPrepareCudaSource::head_dimension(self : Self) -> Int {
  self.head_dimension
}

///|
pub fn DecoderCompressedKvPrepareCudaSource::maximum_position_count(
  self : Self,
) -> Int {
  self.maximum_position_count
}

///|
pub fn DecoderCompressedKvPrepareCudaSource::maximum_compressed_slots(
  self : Self,
) -> Int {
  self.maximum_compressed_slots
}

///|
pub fn DecoderCompressedKvPrepareCudaSource::maximum_key_slots(
  self : Self,
) -> Int {
  self.maximum_key_slots
}

///|
pub fn DecoderCompressedKvPrepareCudaSource::maximum_batches(
  self : Self,
) -> Int {
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

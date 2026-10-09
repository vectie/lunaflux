///|
pub fn DecoderInterleavedRopeCudaSource::operation(
  self : Self,
) -> DecoderInterleavedRopeCudaOperation {
  self.operation
}

///|
pub fn DecoderInterleavedRopeCudaSource::numerical_contract(
  self : Self,
) -> DecoderInterleavedRopeNumericalContract {
  self.numerical_contract
}

///|
pub fn DecoderInterleavedRopeCudaSource::query_head_count(self : Self) -> Int {
  self.query_head_count
}

///|
pub fn DecoderInterleavedRopeCudaSource::key_head_count(self : Self) -> Int {
  self.key_head_count
}

///|
pub fn DecoderInterleavedRopeCudaSource::query_head_dimension(
  self : Self,
) -> Int {
  self.query_head_dimension
}

///|
pub fn DecoderInterleavedRopeCudaSource::rope_dimension(self : Self) -> Int {
  self.rope_dimension
}

///|
pub fn DecoderInterleavedRopeCudaSource::maximum_rows(self : Self) -> Int {
  self.maximum_rows
}

///|
pub fn DecoderInterleavedRopeCudaSource::maximum_positions(self : Self) -> Int {
  self.maximum_positions
}

///|
pub fn DecoderInterleavedRopeCudaSource::theta(self : Self) -> Double {
  self.theta
}

///|
pub fn DecoderInterleavedRopeCudaSource::function_symbol(self : Self) -> String {
  self.function_symbol
}

///|
pub fn DecoderInterleavedRopeCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

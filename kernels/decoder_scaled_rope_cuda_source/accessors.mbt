///|
pub fn DecoderScaledRopeCudaSource::numerical_contract(
  self : Self,
) -> DecoderScaledRopeNumericalContract {
  self.numerical_contract
}

///|
pub fn DecoderScaledRopeCudaSource::query_head_count(self : Self) -> Int {
  self.query_head_count
}

///|
pub fn DecoderScaledRopeCudaSource::key_value_head_count(self : Self) -> Int {
  self.key_value_head_count
}

///|
pub fn DecoderScaledRopeCudaSource::rope_dimension(self : Self) -> Int {
  self.rope_dimension
}

///|
pub fn DecoderScaledRopeCudaSource::maximum_rows(self : Self) -> Int {
  self.maximum_rows
}

///|
pub fn DecoderScaledRopeCudaSource::maximum_positions(self : Self) -> Int {
  self.maximum_positions
}

///|
pub fn DecoderScaledRopeCudaSource::function_symbol(self : Self) -> String {
  self.function_symbol
}

///|
pub fn DecoderScaledRopeCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

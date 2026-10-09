///|
pub fn DecoderInverseRopeCudaSource::numerical_contract(
  self : Self,
) -> DecoderScaledRopeNumericalContract {
  self.numerical_contract
}

///|
pub fn DecoderInverseRopeCudaSource::head_count(self : Self) -> Int {
  self.head_count
}

///|
pub fn DecoderInverseRopeCudaSource::head_dimension(self : Self) -> Int {
  self.head_dimension
}

///|
pub fn DecoderInverseRopeCudaSource::rope_dimension(self : Self) -> Int {
  self.rope_dimension
}

///|
pub fn DecoderInverseRopeCudaSource::maximum_rows(self : Self) -> Int {
  self.maximum_rows
}

///|
pub fn DecoderInverseRopeCudaSource::maximum_positions(self : Self) -> Int {
  self.maximum_positions
}

///|
pub fn DecoderInverseRopeCudaSource::function_symbol(self : Self) -> String {
  self.function_symbol
}

///|
pub fn DecoderInverseRopeCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

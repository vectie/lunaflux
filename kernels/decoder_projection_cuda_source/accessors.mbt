///|
pub fn DecoderProjectionCudaSource::operation(
  self : Self,
) -> DecoderProjectionCudaOperation {
  self.operation
}

///|
pub fn DecoderProjectionCudaSource::numerical_contract(
  self : Self,
) -> DecoderProjectionNumericalContract {
  self.numerical_contract
}

///|
pub fn DecoderProjectionCudaSource::input_width(self : Self) -> Int {
  self.input_width
}

///|
pub fn DecoderProjectionCudaSource::output_width(self : Self) -> Int {
  self.output_width
}

///|
pub fn DecoderProjectionCudaSource::maximum_rows(self : Self) -> Int {
  self.maximum_rows
}

///|
pub fn DecoderProjectionCudaSource::function_symbol(self : Self) -> String {
  self.function_symbol
}

///|
pub fn DecoderProjectionCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn FixedRowsDecoderProjectionCudaSource::operation(
  self : Self,
) -> DecoderProjectionCudaOperation {
  self.operation
}

///|
pub fn FixedRowsDecoderProjectionCudaSource::numerical_contract(
  self : Self,
) -> DecoderProjectionNumericalContract {
  self.numerical_contract
}

///|
pub fn FixedRowsDecoderProjectionCudaSource::row_count(self : Self) -> Int64 {
  self.row_count
}

///|
pub fn FixedRowsDecoderProjectionCudaSource::input_width(self : Self) -> Int {
  self.input_width
}

///|
pub fn FixedRowsDecoderProjectionCudaSource::output_width(self : Self) -> Int {
  self.output_width
}

///|
pub fn FixedRowsDecoderProjectionCudaSource::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn FixedRowsDecoderProjectionCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn GroupedDecoderProjectionCudaSource::operation(
  self : Self,
) -> DecoderProjectionCudaOperation {
  self.operation
}

///|
pub fn GroupedDecoderProjectionCudaSource::numerical_contract(
  self : Self,
) -> DecoderProjectionNumericalContract {
  self.numerical_contract
}

///|
pub fn GroupedDecoderProjectionCudaSource::group_count(self : Self) -> Int {
  self.group_count
}

///|
pub fn GroupedDecoderProjectionCudaSource::input_width(self : Self) -> Int {
  self.input_width
}

///|
pub fn GroupedDecoderProjectionCudaSource::output_width(self : Self) -> Int {
  self.output_width
}

///|
pub fn GroupedDecoderProjectionCudaSource::maximum_rows(self : Self) -> Int {
  self.maximum_rows
}

///|
pub fn GroupedDecoderProjectionCudaSource::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn GroupedDecoderProjectionCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

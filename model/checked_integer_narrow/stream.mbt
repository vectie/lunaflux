///|
/// Incremental startup narrowing with only one row of validation state. No
/// whole I64/I32 host table is retained. The caller supplies bounded output
/// storage and owns all partially published bytes until finish succeeds.
pub struct I64RowNarrowStream {
  priv rows : Int
  priv columns : Int
  priv minimum : Int
  priv maximum : Int
  priv input_bytes : Int64
  priv previous : FixedArray[Int64]
  priv mut received : Int64
  priv mut elements : Int64
  priv mut raw : UInt64
  priv mut partial_bytes : Int
  priv mut failed : Bool
  priv mut finished : Bool
}

///|
pub fn I64RowNarrowStream::new(
  row_count~ : Int,
  column_count~ : Int,
  minimum_value~ : Int,
  exclusive_maximum_value~ : Int,
  maximum_row_state_bytes~ : Int64,
) -> Self raise CheckedIntegerNarrowError {
  guard row_count > 0 &&
    column_count > 0 &&
    exclusive_maximum_value > minimum_value &&
    row_count.to_int64() <= 9223372036854775807L / column_count.to_int64() / 8L &&
    column_count.to_int64() * 8L <= maximum_row_state_bytes else {
    raise InvalidGeometry
  }
  {
    rows: row_count,
    columns: column_count,
    minimum: minimum_value,
    maximum: exclusive_maximum_value,
    input_bytes: row_count.to_int64() * column_count.to_int64() * 8L,
    previous: FixedArray::make(column_count, 0L),
    received: 0L,
    elements: 0L,
    raw: 0UL,
    partial_bytes: 0,
    failed: false,
    finished: false,
  }
}

///|
/// Input may split any I64 element or row. Return the absolute I32 byte offset
/// and live output bytes; the caller must consume them before reusing output.
pub fn I64RowNarrowStream::consume(
  self : Self,
  input_offset : Int64,
  input : ArrayView[Byte],
  output : FixedArray[Byte],
) -> (Int64, Int) raise CheckedIntegerNarrowError {
  guard !self.failed && !self.finished else { raise InvalidStreamState }
  self.failed = true
  guard input_offset == self.received else {
    raise StreamByteCountMismatch(expected=self.received, actual=input_offset)
  }
  guard input.length() > 0 &&
    input.length().to_int64() <= self.input_bytes - self.received else {
    raise StreamByteCountMismatch(
      expected=self.input_bytes - self.received,
      actual=input.length().to_int64(),
    )
  }
  let produced = (input.length().to_int64() + self.partial_bytes.to_int64()) /
    8L
  guard produced <= output.length().to_int64() / 4L else {
    raise OutputByteBudgetExceeded
  }
  let output_offset = self.elements * 4L
  let mut written = 0
  for byte in input {
    self.raw = self.raw | (byte.to_uint64() << (8 * self.partial_bytes))
    self.partial_bytes += 1
    if self.partial_bytes == 8 {
      let value = self.raw.reinterpret_as_int64()
      let row = (self.elements / self.columns.to_int64()).to_int()
      let column = (self.elements % self.columns.to_int64()).to_int()
      require_index_range(value, row, column, self.minimum, self.maximum)
      for previous in 0..<column {
        if self.previous[previous] == value {
          raise DuplicateRowValue(
            row~,
            first_column=previous,
            second_column=column,
            value~,
          )
        }
      }
      self.previous[column] = value
      write_i32_le(output, written, value)
      written += 4
      self.elements += 1L
      self.raw = 0UL
      self.partial_bytes = 0
    }
  }
  self.received += input.length().to_int64()
  self.failed = false
  (output_offset, written)
}

///|
pub fn I64RowNarrowStream::finish(
  self : Self,
) -> Unit raise CheckedIntegerNarrowError {
  guard !self.failed && !self.finished else { raise InvalidStreamState }
  self.failed = true
  guard self.received == self.input_bytes &&
    self.partial_bytes == 0 &&
    self.elements == self.rows.to_int64() * self.columns.to_int64() else {
    raise StreamByteCountMismatch(
      expected=self.input_bytes,
      actual=self.received,
    )
  }
  self.finished = true
  self.failed = false
}

///|
pub fn I64RowNarrowStream::close(self : Self) -> Unit {
  self.previous.fill(0L)
  self.raw = 0UL
  self.partial_bytes = 0
  self.failed = true
}

///|
fn checked_table_elements(
  row_count : Int,
  column_count : Int,
) -> Int raise CheckedIntegerNarrowError {
  if row_count <= 0 ||
    column_count <= 0 ||
    row_count.to_int64() > 2147483647L / column_count.to_int64() {
    raise InvalidGeometry
  }
  row_count * column_count
}

///|
fn decode_i64_le(input : ArrayView[Byte], offset : Int) -> Int64 {
  let mut raw = 0UL
  for shift in 0..<8 {
    raw = raw | (input[offset + shift].to_uint64() << (shift * 8))
  }
  raw.reinterpret_as_int64()
}

///|
fn write_i32_le(output : FixedArray[Byte], offset : Int, value : Int64) -> Unit {
  let raw = value.reinterpret_as_uint64()
  for shift in 0..<4 {
    output[offset + shift] = ((raw >> (shift * 8)) & 0xffUL).to_byte()
  }
}

///|
/// Checked-narrow one exact little-endian row-major I64 table into I32.
/// Validation is completed before the returned owner can escape. Values must
/// lie in `[minimum_value, exclusive_maximum_value)` and each row must contain
/// no duplicate value.
pub fn narrow_unique_i64_le_rows(
  input : ArrayView[Byte],
  row_count~ : Int,
  column_count~ : Int,
  minimum_value~ : Int,
  exclusive_maximum_value~ : Int,
  maximum_output_bytes~ : Int,
) -> CheckedI32Table raise CheckedIntegerNarrowError {
  if minimum_value < -2147483648 ||
    exclusive_maximum_value <= minimum_value ||
    exclusive_maximum_value > 2147483647 ||
    maximum_output_bytes <= 0 {
    raise InvalidGeometry
  }
  let elements = checked_table_elements(row_count, column_count)
  if elements > 2147483647 / 8 {
    raise InvalidGeometry
  }
  let expected_input_bytes = elements * 8
  if input.length() != expected_input_bytes {
    raise InputByteCountMismatch(
      expected=expected_input_bytes,
      actual=input.length(),
    )
  }
  if elements > 2147483647 / 4 || elements * 4 > maximum_output_bytes {
    raise OutputByteBudgetExceeded
  }
  let output = FixedArray::make(elements * 4, b'\x00')
  for row in 0..<row_count {
    for column in 0..<column_count {
      let element = row * column_count + column
      let value = decode_i64_le(input, element * 8)
      if value < minimum_value.to_int64() ||
        value >= exclusive_maximum_value.to_int64() {
        output.fill(b'\x00')
        raise ValueOutOfRange(row~, column~, value~)
      }
      for previous in 0..<column {
        let previous_element = row * column_count + previous
        if decode_i64_le(input, previous_element * 8) == value {
          output.fill(b'\x00')
          raise DuplicateRowValue(
            row~,
            first_column=previous,
            second_column=column,
            value~,
          )
        }
      }
      write_i32_le(output, element * 4, value)
    }
  }
  let digest : CheckedI32TableDigest = {
    hex: @crypto.bytes_to_hex_string(@crypto.sha256(output)),
  }
  {
    row_count,
    column_count,
    minimum_value,
    exclusive_maximum_value,
    digest,
    bytes: Some(output),
    released: false,
  }
}

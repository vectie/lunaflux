///|
fn append_i64_le(output : Array[Byte], value : Int64) -> Unit {
  let raw = value.reinterpret_as_uint64()
  for shift in 0..<8 {
    output.push(((raw >> (shift * 8)) & 0xffUL).to_byte())
  }
}

///|
fn fixture(values : ArrayView[Int64]) -> Array[Byte] {
  let output : Array[Byte] = []
  for value in values {
    append_i64_le(output, value)
  }
  output
}

///|
test "checked I64 rows narrow deterministically and release explicitly" {
  let input = fixture([0L, 2L, 1L, 3L, 5L, 4L])
  let first = narrow_unique_i64_le_rows(
    input,
    row_count=2,
    column_count=3,
    minimum_value=0,
    exclusive_maximum_value=8,
    maximum_output_bytes=24,
  )
  let repeated = narrow_unique_i64_le_rows(
    input,
    row_count=2,
    column_count=3,
    minimum_value=0,
    exclusive_maximum_value=8,
    maximum_output_bytes=24,
  )
  assert_eq(first.row_count(), 2)
  assert_eq(first.column_count(), 3)
  assert_eq(first.byte_count(), 24)
  assert_eq(first.digest(), repeated.digest())
  assert_eq(first.digest().as_hex().length(), 64)
  assert_true(first.release())
  assert_false(first.release())
  assert_true(first.is_released())
  assert_true(repeated.release())
}

///|
test "checked I64 rows reject truncation range duplicates and budget" {
  let valid = fixture([0L, 1L, 2L, 3L])
  try
    narrow_unique_i64_le_rows(
      valid[0:31],
      row_count=2,
      column_count=2,
      minimum_value=0,
      exclusive_maximum_value=4,
      maximum_output_bytes=16,
    )
  catch {
    InputByteCountMismatch(expected=32, actual=31) => ()
    _ => fail("unexpected truncated-input rejection")
  } noraise {
    _ => fail("truncated I64 table was admitted")
  }
  for hostile in [fixture([0L, 4L]), fixture([1L, 1L])] {
    try
      narrow_unique_i64_le_rows(
        hostile,
        row_count=1,
        column_count=2,
        minimum_value=0,
        exclusive_maximum_value=4,
        maximum_output_bytes=8,
      )
    catch {
      ValueOutOfRange(row=0, column=1, value=4L) => ()
      DuplicateRowValue(row=0, first_column=0, second_column=1, value=1L) => ()
      _ => fail("unexpected checked-narrow rejection")
    } noraise {
      _ => fail("invalid I64 row was admitted")
    }
  }
  try
    narrow_unique_i64_le_rows(
      valid,
      row_count=2,
      column_count=2,
      minimum_value=0,
      exclusive_maximum_value=4,
      maximum_output_bytes=15,
    )
  catch {
    OutputByteBudgetExceeded => ()
    _ => fail("unexpected output-budget rejection")
  } noraise {
    _ => fail("oversized derived table was admitted")
  }
}

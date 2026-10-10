///|
test "streamed narrowing is invariant under arbitrary element and row splits" {
  let input = fixture([0L, 2L, 1L, 3L, 5L, 4L])
  for chunk in [1, 3, 7, 8, 9, 16, 23, 64] {
    let stream = I64RowNarrowStream::new(
      row_count=2,
      column_count=3,
      minimum_value=0,
      exclusive_maximum_value=8,
      maximum_row_state_bytes=24L,
    )
    defer stream.close()
    let output = FixedArray::make((chunk + 7) / 8 * 4, b'\xff')
    let complete = FixedArray::make(24, b'\xff')
    for offset = 0; offset < input.length(); offset = offset + chunk {
      let end = (offset + chunk).min(input.length())
      let (start, count) = stream.consume(
        offset.to_int64(),
        input[offset:end],
        output,
      )
      for byte in 0..<count {
        complete[start.to_int() + byte] = output[byte]
      }
    }
    stream.finish()
    for i, value in [0, 2, 1, 3, 5, 4] {
      assert_eq(complete[i * 4], value.to_byte())
      for byte in 1..<4 {
        assert_eq(complete[i * 4 + byte], b'\x00')
      }
    }
    try stream.consume(48L, [b'\x00'], output) catch {
      InvalidStreamState => ()
      e => raise e
    } noraise {
      _ => fail("finished stream accepted more input")
    }
  }
}

///|
test "stream preserves cross-chunk duplicate and range failures" {
  for
    (values, duplicate) in [
      (fixture([1L, 1L]), true),
      (fixture([1L, 8L]), false),
    ] {
    let stream = I64RowNarrowStream::new(
      row_count=1,
      column_count=2,
      minimum_value=0,
      exclusive_maximum_value=8,
      maximum_row_state_bytes=16L,
    )
    let output = FixedArray::make(8, b'\x00')
    ignore(stream.consume(0L, values[:9], output))
    try stream.consume(9L, values[9:], output) catch {
      DuplicateRowValue(row=0, first_column=0, second_column=1, value=1L) if duplicate =>
        ()
      ValueOutOfRange(row=0, column=1, value=8L) if !duplicate => ()
      e => raise e
    } noraise {
      _ => fail("invalid streamed row accepted")
    }
    try stream.finish() catch {
      InvalidStreamState => ()
      e => raise e
    } noraise {
      _ => fail("failed stream became a finished table")
    }
    stream.close()
    stream.close()
  }
}

///|
test "stream rejects truncation offset discontinuity and unbudgeted row state" {
  let stream = I64RowNarrowStream::new(
    row_count=1,
    column_count=2,
    minimum_value=-4,
    exclusive_maximum_value=8,
    maximum_row_state_bytes=16L,
  )
  defer stream.close()
  let input = fixture([-4L, 7L])
  let output = FixedArray::make(8, b'\x00')
  ignore(stream.consume(0L, input[:15], output))
  try stream.finish() catch {
    StreamByteCountMismatch(expected=16L, actual=15L) => ()
    e => raise e
  } noraise {
    _ => fail("partial I64 accepted")
  }
  let gap = I64RowNarrowStream::new(
    row_count=1,
    column_count=2,
    minimum_value=0,
    exclusive_maximum_value=8,
    maximum_row_state_bytes=16L,
  )
  defer gap.close()
  try gap.consume(1L, [b'\x00'], output) catch {
    StreamByteCountMismatch(expected=0L, actual=1L) => ()
    e => raise e
  } noraise {
    _ => fail("noncontiguous source accepted")
  }
  try
    I64RowNarrowStream::new(
      row_count=1,
      column_count=2,
      minimum_value=0,
      exclusive_maximum_value=8,
      maximum_row_state_bytes=15L,
    )
  catch {
    InvalidGeometry => ()
    e => raise e
  } noraise {
    _ => fail("unbudgeted row validation state allocated")
  }
}

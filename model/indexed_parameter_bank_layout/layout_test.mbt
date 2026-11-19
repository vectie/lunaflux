///|
fn slice(
  index : Int32,
  ordinal : Int32,
  name : String,
  bytes : Int64,
) -> @indexed_parameter_bank_layout.IndexedParameterSliceInput {
  try! @indexed_parameter_bank_layout.IndexedParameterSliceInput::new(
    logical_index=index,
    source_ordinal=ordinal,
    source_name=name,
    byte_count=bytes,
  )
}

///|
test "indexed bank layout binds exact source order and checked offsets" {
  let layout = @indexed_parameter_bank_layout.build_indexed_parameter_bank_layout(
    [slice(0, 7, "expert.0", 16L), slice(1, 9, "expert.1", 32L)],
    alignment=16L,
    maximum_byte_count=48L,
  )
  assert_eq(layout.byte_count(), 48L)
  assert_eq(layout.slices()[0].target_byte_offset(), 0L)
  assert_eq(layout.slices()[1].target_byte_offset(), 16L)
  assert_eq(layout.slices()[1].source_ordinal(), 9)
}

///|
test "indexed bank layout rejects duplicate reorder alignment extent and overflow" {
  let cases = [
    [slice(0, 7, "a", 16L), slice(1, 7, "b", 16L)],
    [slice(0, 7, "a", 16L), slice(1, 8, "a", 16L)],
    [slice(1, 7, "a", 16L)],
    [slice(0, 7, "a", 15L)],
  ]
  for inputs in cases {
    try
      @indexed_parameter_bank_layout.build_indexed_parameter_bank_layout(
        inputs,
        alignment=16L,
        maximum_byte_count=64L,
      )
    catch {
      DuplicateSourceOrdinal(_)
      | DuplicateSourceName(_)
      | UnexpectedLogicalIndex(_)
      | InvalidExtent => ()
      _ => fail("wrong indexed layout error")
    } noraise {
      _ => fail("invalid indexed layout admitted")
    }
  }
  try
    @indexed_parameter_bank_layout.build_indexed_parameter_bank_layout(
      [slice(0, 7, "a", 32L), slice(1, 8, "b", 32L)],
      alignment=16L,
      maximum_byte_count=63L,
    )
  catch {
    OffsetOverflow => ()
    _ => fail("wrong overflow error")
  } noraise {
    _ => fail("overflowing indexed layout admitted")
  }
  try
    @indexed_parameter_bank_layout.build_indexed_parameter_bank_layout(
      [slice(0, 7, "a", 16L)],
      alignment=3L,
      maximum_byte_count=16L,
    )
  catch {
    InvalidAlignment => ()
    _ => fail("wrong alignment error")
  } noraise {
    _ => fail("invalid alignment admitted")
  }
}

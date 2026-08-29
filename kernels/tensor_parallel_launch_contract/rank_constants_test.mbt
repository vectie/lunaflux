///|
fn read_i32(bytes : FixedArray[Byte], offset : Int) -> Int {
  (bytes[offset].to_uint() |
  (bytes[offset + 1].to_uint() << 8) |
  (bytes[offset + 2].to_uint() << 16) |
  (bytes[offset + 3].to_uint() << 24)).reinterpret_as_int()
}

///|
test "rank constants encode exact little-endian v2 rank runtime cells" {
  let source = fixture()
  let profile = fixture_profile()
  let set = admit_fixture(
    source,
    profile,
    fixture_inputs(source.model, source.rank_plan, source.target, profile),
  )
  let constants = set.rank_constants()
  let bytes = FixedArray::make(38, b'\xff')
  constants.encode_into(bytes, destination_offset=3)
  assert_eq(bytes[0], b'\xff')
  assert_eq(bytes[2], b'\xff')
  assert_eq(bytes[35], b'\xff')
  assert_eq(bytes[37], b'\xff')
  let expected : Array[Int] = [2, 1, 2, 8, 0, 1, 64, 0]
  for index, value in expected {
    assert_eq(read_i32(bytes, 3 + index * 4), value)
  }
  assert_eq(bytes[3], b'\x02')
  assert_eq(bytes[4], b'\x00')
  assert_eq(bytes[31], b'\x00')
  assert_eq(bytes[34], b'\x00')
}

///|
test "rank constants reject negative and truncated destinations without writes" {
  let source = fixture()
  let profile = fixture_profile()
  let constants = admit_fixture(
    source,
    profile,
    fixture_inputs(source.model, source.rank_plan, source.target, profile),
  ).rank_constants()
  for pair in [(-1, 32), (0, 31), (1, 32), (33, 32)] {
    let (offset, length) = pair
    let bytes = FixedArray::make(length, b'\x5a')
    expect_rule(
      () => constants.encode_into(bytes, destination_offset=offset),
      RankConstantsDestination,
    )
    for byte in bytes {
      assert_eq(byte, b'\x5a')
    }
  }
}

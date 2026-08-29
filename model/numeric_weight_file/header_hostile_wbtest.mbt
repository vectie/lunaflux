///|
fn find_fragment(file : Bytes, fragment : Bytes) -> Int {
  for start in 8..<=(file.length() - fragment.length()) {
    let mut matches = true
    for index in 0..<fragment.length() {
      if file[start + index] != fragment[index] {
        matches = false
      }
    }
    if matches {
      return start
    }
  }
  abort("fixture fragment missing")
}

///|
fn fixture_header_length(file : Bytes) -> Int {
  let mut length = 0UL
  for index in 0..<8 {
    length = length | (file[index].to_uint64() << (index * 8))
  }
  length.to_int()
}

///|
fn replace_header_fragment(
  file : Bytes,
  needle : String,
  replacement : String,
) -> Bytes {
  let encoded_needle = @utf8.encode(needle)
  let encoded_replacement = @utf8.encode(replacement)
  let position = find_fragment(file, encoded_needle)
  let old_header_length = fixture_header_length(file)
  let new_header_length = old_header_length -
    encoded_needle.length() +
    encoded_replacement.length()
  let output : Array[Byte] = []
  append_u64_le(output, new_header_length.to_uint64())
  for index in 8..<position {
    output.push(file[index])
  }
  for byte in encoded_replacement {
    output.push(byte)
  }
  for index in (position + encoded_needle.length())..<file.length() {
    output.push(file[index])
  }
  Bytes::from_array(output)
}

///|
fn append_root_header_field(file : Bytes, field : String) -> Bytes {
  let old_header_length = fixture_header_length(file)
  let encoded = @utf8.encode(field)
  let root_close = 8 + old_header_length - 1
  let output : Array[Byte] = []
  append_u64_le(output, (old_header_length + encoded.length()).to_uint64())
  for index in 8..<root_close {
    output.push(file[index])
  }
  for byte in encoded {
    output.push(byte)
  }
  for index in root_close..<file.length() {
    output.push(file[index])
  }
  Bytes::from_array(output)
}

///|
fn expect_fixture_failure(
  plan : @plan.ModelPlan,
  file : Bytes,
  expected : NumericWeightFileError,
) -> Unit raise {
  try prepare_fixture(plan, file) catch {
    error => assert_eq(error, expected)
  } noraise {
    _ => fail("hostile header admitted")
  }
}

///|
test "header rejects duplicate unknown missing and aliased fields and names" {
  let plan = numeric_fixture_plan()
  let valid = numeric_fixture_file(plan)
  expect_fixture_failure(
    plan,
    replace_header_fragment(
      valid, "\"dtype\":\"I8\"", "\"dtype\":\"I8\",\"dtype\":\"I8\"",
    ),
    DuplicateHeaderKey,
  )
  expect_fixture_failure(
    plan,
    replace_header_fragment(
      valid, "lunaflux.tensor.v1.1", "lunaflux.tensor.v1.9",
    ),
    MissingTensor(ordinal=1),
  )
  expect_fixture_failure(
    plan,
    replace_header_fragment(
      valid, "lunaflux.tensor.v1.1", "lunaflux.tensor.v1.01",
    ),
    MissingTensor(ordinal=1),
  )
  expect_fixture_failure(
    plan,
    replace_header_fragment(valid, "\"dtype\":\"I8\"", "\"dtyqe\":\"I8\""),
    UnexpectedTensorField(ordinal=1),
  )
  expect_fixture_failure(
    plan,
    replace_header_fragment(valid, "\"dtype\":\"I8\",", ""),
    UnexpectedTensorField(ordinal=1),
  )
  let extra = ",\"unknown\":{\"dtype\":\"I8\",\"shape\":[1],\"data_offsets\":[21,22]}"
  expect_fixture_failure(
    plan,
    append_root_header_field(valid, extra),
    TensorCountMismatch(expected=3, actual=4),
  )
}

///|
test "header rejects overlap and gap at the first divergent ordinal" {
  let plan = numeric_fixture_plan()
  let valid = numeric_fixture_file(plan)
  expect_fixture_failure(
    plan,
    replace_header_fragment(valid, "[16,17]", "[15,16]"),
    PayloadNotContiguous(ordinal=1),
  )
  expect_fixture_failure(
    plan,
    replace_header_fragment(valid, "[16,17]", "[17,18]"),
    PayloadNotContiguous(ordinal=1),
  )
}

///|
test "declared header bound precedes schema and layout preflight without body read" {
  let plan = numeric_fixture_plan()
  let prefix : Array[Byte] = []
  append_u64_le(prefix, 1024UL * 1024UL)
  let maximum_read = Ref(0)
  let read = (
    destination : FixedArray[Byte],
    destination_offset : Int,
    _ : Int64,
    byte_count : Int,
  ) => {
    if byte_count < 0 {
      raise FileTruncated
    }
    maximum_read.val = maximum_read.val.max(byte_count)
    for index in 0..<byte_count {
      destination[destination_offset + index] = if index < 8 {
        prefix[index]
      } else {
        b'\x00'
      }
    }
  }
  try
    prepare_numeric_weight_file(
      plan,
      try! NumericWeightArtifactDigest::from_sha256(
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      ),
      1,
      2L * 1024L * 1024L,
      try! NumericWeightFileLimits::new(
        max_file_bytes=2L * 1024L * 1024L,
        max_header_bytes=1024 * 1024,
        transfer_chunk_bytes=16,
        max_tensor_count=2,
      ),
      16L,
      4096L,
      read,
      () => 1,
    )
  catch {
    InvalidLimit(TensorCount) => ()
    _ => fail("unexpected preflight error")
  } noraise {
    _ => fail("schema limit admitted")
  }
  assert_eq(maximum_read.val, 8)
  maximum_read.val = 0
  try
    prepare_numeric_weight_file(
      plan,
      try! NumericWeightArtifactDigest::from_sha256(
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      ),
      1,
      2L * 1024L * 1024L,
      try! NumericWeightFileLimits::new(
        max_file_bytes=2L * 1024L * 1024L,
        max_header_bytes=1024 * 1024,
        transfer_chunk_bytes=16,
        max_tensor_count=16,
      ),
      3L,
      4096L,
      read,
      () => 1,
    )
  catch {
    InvalidLayout(InvalidAlignment) => ()
    _ => fail("unexpected preflight error")
  } noraise {
    _ => fail("invalid alignment admitted")
  }
  assert_eq(maximum_read.val, 8)
}

///|
test "second pass rejects same-stamp byte mutation by digest and closes arena" {
  let plan = numeric_fixture_plan()
  let file = numeric_fixture_file(plan)
  let prepared = prepare_fixture(plan, file)
  let owner : Ref[FixtureArena?] = Ref(None)
  try
    transfer_prepared_numeric_file(
      prepared,
      (destination, destination_offset, position, byte_count) => {
        for index in 0..<byte_count {
          let absolute = position.to_int() + index
          destination[destination_offset + index] = if absolute ==
            file.length() - 1 {
            file[absolute] ^ b'\x01'
          } else {
            file[absolute]
          }
        }
      },
      () => 1,
      arena_bytes => {
        let arena = {
          bytes: FixedArray::make(arena_bytes.to_int(), b'\x00'),
          closed: false,
        }
        owner.val = Some(arena)
        arena
      },
      (arena, source, source_offset, destination_offset, byte_count) => {
        for index in 0L..<byte_count {
          arena.bytes[(destination_offset + index).to_int()] = source[(source_offset.to_int64() +
            index).to_int()]
        }
      },
      arena => arena.closed = true,
    )
  catch {
    ArtifactDigestMismatch => ()
    _ => fail("unexpected error")
  } noraise {
    _ => fail("same-stamp mutation admitted")
  }
  guard owner.val is Some(arena) else { fail("arena was not opened") }
  assert_true(arena.closed)
}

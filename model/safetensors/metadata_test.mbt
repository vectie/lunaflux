///|
fn limits(
  max_materialized_tensor_bytes? : Int64 = 1024L,
) -> @safetensors.HeaderLimits raise {
  @safetensors.HeaderLimits::new(
    max_header_bytes=4096,
    max_tensor_count=16,
    max_rank=8,
    max_name_length=128,
    max_payload_bytes=4096L,
    max_materialized_tensor_bytes~,
  )
}

///|
fn raw(
  name : String,
  shape : ArrayView[Int64],
  start : Int64,
  end : Int64,
  dtype? : String = "BF16",
) -> @safetensors.RawTensorMetadata {
  @safetensors.RawTensorMetadata::new(
    name~,
    dtype~,
    shape~,
    data_start=start,
    data_end=end,
  )
}

///|
test "valid bounded metadata exposes immutable ranges and total size" {
  let entries = [
    raw("model.embed.weight", [2L, 4L], 0L, 16L),
    raw("model.norm.weight", [2L, 2L], 16L, 24L),
  ]
  let metadata = @safetensors.validate_metadata(
    entries,
    encoded_header_bytes=256,
    payload_bytes=24L,
    limits=limits(),
  )
  assert_eq(metadata.tensor_count(), 2)
  assert_eq(metadata.materialized_tensor_bytes(), 24L)
  guard metadata.find_tensor("model.norm.weight") is Some(norm) else {
    fail("expected tensor")
  }
  assert_eq(norm.data_start(), 16L)
  assert_eq(norm.byte_length(), 8L)
}

///|
test "malformed dtype and byte ranges are rejected" {
  try
    @safetensors.validate_metadata(
      [raw("weight", [2L], 0L, 4L, dtype="F16")],
      encoded_header_bytes=64,
      payload_bytes=4L,
      limits=limits(),
    )
  catch {
    @safetensors.SafetensorsError::InvalidDType(0) => ()
    _ => fail("unexpected metadata error")
  } noraise {
    _ => fail("expected invalid dtype")
  }
  try
    @safetensors.validate_metadata(
      [raw("weight", [2L], 4L, 3L)],
      encoded_header_bytes=64,
      payload_bytes=4L,
      limits=limits(),
    )
  catch {
    @safetensors.SafetensorsError::InvalidByteRange(0) => ()
    _ => fail("unexpected metadata error")
  } noraise {
    _ => fail("expected invalid range")
  }
}

///|
test "duplicate and overlapping tensor ranges are rejected" {
  try
    @safetensors.validate_metadata(
      [raw("weight", [2L, 2L], 0L, 8L), raw("weight", [2L], 8L, 12L)],
      encoded_header_bytes=64,
      payload_bytes=12L,
      limits=limits(),
    )
  catch {
    @safetensors.SafetensorsError::DuplicateTensor(0, 1) => ()
    _ => fail("unexpected metadata error")
  } noraise {
    _ => fail("expected duplicate tensor")
  }
  try
    @safetensors.validate_metadata(
      [raw("left", [2L, 2L], 0L, 8L), raw("right", [2L, 2L], 4L, 12L)],
      encoded_header_bytes=64,
      payload_bytes=12L,
      limits=limits(),
    )
  catch {
    @safetensors.SafetensorsError::OverlappingTensors(0, 1) => ()
    _ => fail("unexpected metadata error")
  } noraise {
    _ => fail("expected overlap")
  }
}

///|
test "shape multiplication overflow and payload bounds are rejected" {
  try
    @safetensors.validate_metadata(
      [raw("huge", [9223372036854775807L, 2L], 0L, 2L)],
      encoded_header_bytes=64,
      payload_bytes=2L,
      limits=limits(),
    )
  catch {
    @safetensors.SafetensorsError::TensorSizeOverflow(0) => ()
    _ => fail("unexpected metadata error")
  } noraise {
    _ => fail("expected overflow")
  }
  try
    @safetensors.validate_metadata(
      [raw("weight", [2L], 2L, 6L)],
      encoded_header_bytes=64,
      payload_bytes=4L,
      limits=limits(),
    )
  catch {
    @safetensors.SafetensorsError::RangeOutsidePayload(0) => ()
    _ => fail("unexpected metadata error")
  } noraise {
    _ => fail("expected out-of-bounds range")
  }
}

///|
test "materialization total is checked independently of payload size" {
  try
    @safetensors.validate_metadata(
      [raw("left", [2L, 2L], 0L, 8L), raw("right", [2L, 2L], 8L, 16L)],
      encoded_header_bytes=64,
      payload_bytes=16L,
      limits=limits(max_materialized_tensor_bytes=12L),
    )
  catch {
    @safetensors.SafetensorsError::MaterializedSizeExceeded => ()
    _ => fail("unexpected metadata error")
  } noraise {
    _ => fail("expected materialization limit failure")
  }
}

///|
test "header and tensor-count envelopes fail before entry validation" {
  try
    @safetensors.validate_metadata(
      [],
      encoded_header_bytes=4097,
      payload_bytes=1L,
      limits=limits(),
    )
  catch {
    @safetensors.SafetensorsError::HeaderTooLarge => ()
    _ => fail("unexpected metadata error")
  } noraise {
    _ => fail("expected header bound failure")
  }
}

///|
fn limits(
  max_header_bytes? : Int = 4096,
  max_tensor_count? : Int = 8,
  max_payload_bytes? : Int64 = 4096L,
) -> @safetensors.HeaderLimits raise {
  @safetensors.HeaderLimits::new(
    max_header_bytes~,
    max_tensor_count~,
    max_rank=8,
    max_name_length=128,
    max_payload_bytes~,
    max_materialized_tensor_bytes=max_payload_bytes,
  )
}

///|
fn fixture_with_length(
  header : StringView,
  encoded_length : UInt64,
  payload_bytes? : Int = 0,
) -> Bytes {
  let bytes : Array[Byte] = []
  for shift in 0..<8 {
    bytes.push((encoded_length >> (shift * 8)).to_byte())
  }
  for byte in @utf8.encode(header) {
    bytes.push(byte)
  }
  for _ in 0..<payload_bytes {
    bytes.push(0)
  }
  Bytes::from_array(bytes)
}

///|
fn fixture(header : StringView, payload_bytes? : Int = 0) -> Bytes {
  fixture_with_length(
    header,
    @utf8.encode(header).length().to_uint64(),
    payload_bytes~,
  )
}

///|
test "valid header metadata is decoded and payload-relative ranges are preserved" {
  let encoded_header =
    #|{"__metadata__":{"format":"pt"},"model.embed.weight":{"dtype":"BF16","shape":[2,4],"data_offsets":[0,16]},"model.norm.weight":{"dtype":"BF16","shape":[2],"data_offsets":[16,20]}}
  let header = String::add(encoded_header, "   ")
  let file = fixture(header, payload_bytes=20)
  let metadata = @safetensors_reader.read_metadata(file, limits=limits())
  assert_eq(metadata.tensor_count(), 2)
  assert_eq(metadata.encoded_header_bytes(), 181)
  assert_eq(metadata.payload_bytes(), 20L)
  guard metadata.find_tensor("model.norm.weight") is Some(norm) else {
    fail("expected tensor")
  }
  assert_eq(norm.data_start(), 16L)
  assert_eq(norm.data_end(), 20L)
}

///|
test "exact prefix and header validate without owning the payload" {
  let header = "{\"w\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,4]}}"
  let prefix_and_header = fixture(header)
  let metadata = @safetensors_reader.read_header(
    prefix_and_header,
    payload_bytes=4L,
    limits=limits(),
  )
  assert_eq(metadata.tensor_count(), 1)
  assert_eq(metadata.payload_bytes(), 4L)
  try
    @safetensors_reader.read_header(
      fixture(header, payload_bytes=1),
      payload_bytes=4L,
      limits=limits(),
    )
  catch {
    @safetensors_reader.SafetensorsReaderError::HeaderEnvelopeSizeMismatch => ()
    _ => fail("unexpected header-envelope error")
  } noraise {
    _ => fail("expected extra header bytes to be rejected")
  }
  try
    @safetensors_reader.read_header(
      prefix_and_header,
      payload_bytes=-1L,
      limits=limits(),
    )
  catch {
    @safetensors_reader.SafetensorsReaderError::InvalidPayloadLength => ()
    _ => fail("unexpected payload-length error")
  } noraise {
    _ => fail("expected negative payload length rejection")
  }
}

///|
test "declared header length is bounded from an exact prefix" {
  let encoded = fixture_with_length("", 37UL)
  assert_eq(
    @safetensors_reader.declared_header_bytes(encoded, limits=limits()),
    37,
  )
  try
    @safetensors_reader.declared_header_bytes(
      fixture_with_length("x", 1UL),
      limits=limits(),
    )
  catch {
    @safetensors_reader.SafetensorsReaderError::HeaderEnvelopeSizeMismatch => ()
    _ => fail("unexpected prefix-size error")
  } noraise {
    _ => fail("expected extra prefix bytes to fail")
  }
  try
    @safetensors_reader.declared_header_bytes(
      fixture_with_length("", 4097UL),
      limits=limits(),
    )
  catch {
    @safetensors_reader.SafetensorsReaderError::HeaderTooLarge => ()
    _ => fail("unexpected prefix-limit error")
  } noraise {
    _ => fail("expected oversized declaration to fail")
  }
}

///|
test "missing overflowing oversized and truncated header lengths are rejected" {
  try @safetensors_reader.read_metadata(b"short", limits=limits()) catch {
    @safetensors_reader.SafetensorsReaderError::MissingHeaderLength => ()
    _ => fail("unexpected reader error")
  } noraise {
    _ => fail("expected missing prefix")
  }
  try
    @safetensors_reader.read_metadata(
      fixture_with_length("{}", 2147483648UL),
      limits=limits(),
    )
  catch {
    @safetensors_reader.SafetensorsReaderError::HeaderLengthOverflow => ()
    _ => fail("unexpected reader error")
  } noraise {
    _ => fail("expected overflowing length")
  }
  try
    @safetensors_reader.read_metadata(
      fixture_with_length("{}", 2UL),
      limits=limits(max_header_bytes=1),
    )
  catch {
    @safetensors_reader.SafetensorsReaderError::HeaderTooLarge => ()
    _ => fail("unexpected reader error")
  } noraise {
    _ => fail("expected header limit")
  }
  try
    @safetensors_reader.read_metadata(
      fixture_with_length("{}", 3UL),
      limits=limits(),
    )
  catch {
    @safetensors_reader.SafetensorsReaderError::TruncatedHeader => ()
    _ => fail("unexpected reader error")
  } noraise {
    _ => fail("expected truncation")
  }
}

///|
test "header envelope encoding JSON root and metadata are strict" {
  let invalid : Array[(Bytes, @safetensors_reader.SafetensorsReaderError)] = [
    (fixture(" {}"), InvalidHeaderEnvelope),
    (fixture("{}\n"), InvalidHeaderEnvelope),
    (fixture("{\"x\":}"), InvalidHeaderJson),
    (fixture("[]"), InvalidHeaderEnvelope),
    (fixture("{\"__metadata__\":[]}"), InvalidMetadata),
    (fixture("{\"__metadata__\":{\"format\":1}}"), InvalidMetadata),
  ]
  for item in invalid {
    let (file, expected) = item
    try @safetensors_reader.read_metadata(file, limits=limits()) catch {
      error => assert_eq(@debug.to_string(error), @debug.to_string(expected))
    } noraise {
      _ => fail("expected invalid header")
    }
  }
  let invalid_utf8 : Bytes = [2, 0, 0, 0, 0, 0, 0, 0, b'{', 0xff]
  try @safetensors_reader.read_metadata(invalid_utf8, limits=limits()) catch {
    @safetensors_reader.SafetensorsReaderError::InvalidHeaderEnvelope => ()
    _ => fail("unexpected reader error")
  } noraise {
    _ => fail("expected invalid envelope before decoding")
  }
  let invalid_utf8 : Bytes = [3, 0, 0, 0, 0, 0, 0, 0, b'{', 0xff, b'}']
  try @safetensors_reader.read_metadata(invalid_utf8, limits=limits()) catch {
    @safetensors_reader.SafetensorsReaderError::InvalidHeaderEncoding => ()
    _ => fail("unexpected reader error")
  } noraise {
    _ => fail("expected invalid UTF-8")
  }
}

///|
test "payload envelope is checked before header decoding and JSON parsing" {
  try
    @safetensors_reader.read_metadata(
      fixture("{\"x\":}", payload_bytes=5),
      limits=limits(max_payload_bytes=4L),
    )
  catch {
    @safetensors_reader.SafetensorsReaderError::Validation(
      RangeOutsidePayload(0)
    ) => ()
    _ => fail("unexpected reader error")
  } noraise {
    _ => fail("expected payload limit before malformed JSON")
  }
}

///|
test "duplicate tensor names and fields are rejected before map overwrite" {
  let duplicate_headers : Array[(String, Int)] = [
    (
      "{\"w\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]},\"w\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}}",
      1,
    ),
    (
      "{\"w\":{\"dtype\":\"BF16\",\"dtype\":\"F16\",\"shape\":[1],\"data_offsets\":[0,2]}}",
      2,
    ),
    (
      "{\"w\":{\"dtype\":\"BF16\",\"sh\\u0061pe\":[1],\"shape\":[1],\"data_offsets\":[0,2]}}",
      2,
    ),
  ]
  for item in duplicate_headers {
    let (header, expected_depth) = item
    try
      @safetensors_reader.read_metadata(
        fixture(header, payload_bytes=2),
        limits=limits(),
      )
    catch {
      @safetensors_reader.SafetensorsReaderError::DuplicateHeaderKey(
        depth~,
        index=_
      ) => assert_eq(depth, expected_depth)
      _ => fail("unexpected reader error")
    } noraise {
      _ => fail("expected duplicate-key rejection")
    }
  }
}

///|
test "exact Int64 metadata numbers accept boundaries and integral JSON values" {
  let exact_integer_headers : Array[String] = [
    "{\"w\":{\"dtype\":\"BF16\",\"shape\":[1.0],\"data_offsets\":[0e0,2e0]}}", "{\"w\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}}",
  ]
  for header in exact_integer_headers {
    let metadata = @safetensors_reader.read_metadata(
      fixture(header, payload_bytes=2),
      limits=limits(),
    )
    assert_eq(metadata.materialized_tensor_bytes(), 2L)
  }
  try
    @safetensors_reader.read_metadata(
      fixture(
        "{\"w\":{\"dtype\":\"BF16\",\"shape\":[9223372036854775807],\"data_offsets\":[0,2]}}",
        payload_bytes=2,
      ),
      limits=limits(),
    )
  catch {
    @safetensors_reader.SafetensorsReaderError::Validation(
      TensorSizeOverflow(0)
    ) => ()
    _ => fail("unexpected reader error")
  } noraise {
    _ => fail("expected exact Int64 max to reach size validation")
  }
}

///|
test "tensor object fields types and exact integers are rejected safely" {
  let malformed : Array[(String, @safetensors_reader.SafetensorsReaderError)] = [
    ("{\"w\":[]}", InvalidTensor(0)),
    (
      "{\"w\":{\"shape\":[2],\"data_offsets\":[0,4]}}",
      MissingTensorField(0, DType),
    ),
    (
      "{\"w\":{\"dtype\":1,\"shape\":[2],\"data_offsets\":[0,4]}}",
      InvalidTensorField(0, DType),
    ),
    (
      "{\"w\":{\"dtype\":\"BF16\",\"shape\":{},\"data_offsets\":[0,4]}}",
      InvalidTensorField(0, Shape),
    ),
    (
      "{\"w\":{\"dtype\":\"BF16\",\"shape\":[2.5],\"data_offsets\":[0,4]}}",
      InvalidInteger(0, Shape),
    ),
    (
      "{\"w\":{\"dtype\":\"BF16\",\"shape\":[9223372036854775808],\"data_offsets\":[0,4]}}",
      InvalidInteger(0, Shape),
    ),
    (
      "{\"w\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0]}}",
      InvalidTensorField(0, DataOffsets),
    ),
    (
      "{\"w\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,4],\"extra\":0}}",
      UnexpectedTensorField(0),
    ),
  ]
  for item in malformed {
    let (header, expected) = item
    try
      @safetensors_reader.read_metadata(
        fixture(header, payload_bytes=4),
        limits=limits(),
      )
    catch {
      error => assert_eq(@debug.to_string(error), @debug.to_string(expected))
    } noraise {
      _ => fail("expected malformed tensor")
    }
  }
}

///|
test "validation and complete-payload failures remain typed and payload-safe" {
  try
    @safetensors_reader.read_metadata(
      fixture(
        "{\"w\":{\"dtype\":\"F16\",\"shape\":[2],\"data_offsets\":[0,4]}}",
        payload_bytes=4,
      ),
      limits=limits(),
    )
  catch {
    @safetensors_reader.SafetensorsReaderError::Validation(InvalidDType(0)) =>
      ()
    _ => fail("unexpected reader error")
  } noraise {
    _ => fail("expected dtype validation")
  }
  let invalid_ranges : Array[(String, @safetensors.SafetensorsError)] = [
    (
      "{\"w\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[-1,3]}}",
      InvalidByteRange(0),
    ),
    (
      "{\"w\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[2,6]}}",
      RangeOutsidePayload(0),
    ),
    (
      "{\"w\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,2]}}",
      RangeSizeMismatch(0),
    ),
  ]
  for item in invalid_ranges {
    let (header, expected) = item
    try
      @safetensors_reader.read_metadata(
        fixture(header, payload_bytes=4),
        limits=limits(),
      )
    catch {
      @safetensors_reader.SafetensorsReaderError::Validation(error) =>
        assert_eq(error, expected)
      _ => fail("unexpected reader error")
    } noraise {
      _ => fail("expected range validation")
    }
  }
  try
    @safetensors_reader.read_metadata(
      fixture(
        "{\"w\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,4]}}",
        payload_bytes=6,
      ),
      limits=limits(),
    )
  catch {
    @safetensors_reader.SafetensorsReaderError::PayloadSizeMismatch => ()
    _ => fail("unexpected reader error")
  } noraise {
    _ => fail("expected trailing payload rejection")
  }
  try
    @safetensors_reader.read_metadata(
      fixture(
        "{\"left\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]},\"right\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[4,6]}}",
        payload_bytes=6,
      ),
      limits=limits(),
    )
  catch {
    @safetensors_reader.SafetensorsReaderError::PayloadSizeMismatch => ()
    _ => fail("unexpected reader error")
  } noraise {
    _ => fail("expected payload hole rejection")
  }
  try
    @safetensors_reader.read_metadata(
      fixture_with_length("{} ", 2UL),
      limits=limits(),
    )
  catch {
    @safetensors_reader.SafetensorsReaderError::PayloadSizeMismatch => ()
    _ => fail("unexpected reader error")
  } noraise {
    _ => fail("expected bytes beyond the declared header to be payload")
  }
}

///|
test "tensor count is bounded before decoded entry accumulation grows" {
  try
    @safetensors_reader.read_metadata(
      fixture(
        "{\"a\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]},\"b\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[2,4]}}",
        payload_bytes=4,
      ),
      limits=limits(max_tensor_count=1),
    )
  catch {
    @safetensors_reader.SafetensorsReaderError::Validation(TooManyTensors) => ()
    _ => fail("unexpected reader error")
  } noraise {
    _ => fail("expected tensor-count limit")
  }
}

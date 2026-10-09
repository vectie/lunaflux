///|
fn payload_test_content(fill? : Char = 'a') -> @spec.ContentDigest {
  try! @spec.ContentDigest::from_sha256(String::make(64, fill))
}

///|
fn payload_test_limits(
  maximum_elements? : Int64 = 200000L,
  maximum_payload_bytes? : Int64 = 200000L,
  maximum_canonical_bytes? : Int = 4096,
) -> DeepSeekV4QuantizedPayloadLimits {
  try! DeepSeekV4QuantizedPayloadLimits::new(
    maximum_elements~,
    maximum_payload_bytes~,
    maximum_canonical_bytes~,
  )
}

///|
fn payload_test_block(
  storage : @deepseek_v4_numeric_plan.DeepSeekV4LogicalStorage,
) -> DeepSeekV4QuantizedBlockShape {
  match storage {
    Fp8E4M3Block128Parameter | Ue8M0Block128ScaleMetadata =>
      try! DeepSeekV4QuantizedBlockShape::new(rows=128, columns=128)
    PackedFp4E2M1Block32Parameter | Ue8M0Block32ScaleMetadata =>
      try! DeepSeekV4QuantizedBlockShape::new(rows=1, columns=32)
    _ => abort("plain storage has no quantized payload block")
  }
}

///|
fn payload_test_admit(
  storage : @deepseek_v4_numeric_plan.DeepSeekV4LogicalStorage,
  shape : ArrayView[Int64],
  bytes : Bytes,
  fill? : Char = 'a',
) -> DeepSeekV4QuantizedPayloadContract raise DeepSeekV4QuantizedPayloadError {
  admit(
    storage~,
    block_shape=payload_test_block(storage),
    logical_shape=shape,
    model_content_digest=payload_test_content(fill~),
    payload_bytes=bytes,
    limits=payload_test_limits(),
  )
}

///|
test "independent finite E4M3 bit patterns decode exactly" {
  let cases : Array[(Byte, Double)] = [
    (b'\x00', 0.0),
    (b'\x01', 0.001953125),
    (b'\x08', 0.015625),
    (b'\x38', 1.0),
    (b'\x3c', 1.5),
    (b'\x40', 2.0),
    (b'\x77', 240.0),
    (b'\x78', 256.0),
    (b'\x7e', 448.0),
    (b'\xb8', -1.0),
  ]
  for item in cases {
    let (code, expected) = item
    assert_eq(decode_finite_fp8_e4m3(code), expected)
  }
  for code in [b'\x7f', b'\xff'] {
    try decode_finite_fp8_e4m3(code) catch {
      NonFiniteFp8Code(actual) => assert_eq(actual, code)
      _ => fail("unexpected non-finite E4M3 rejection")
    } noraise {
      _ => fail("E4M3FN NaN code decoded as finite")
    }
  }
}

///|
test "FP8 payload identity binds encoding shape content and bytes" {
  let bytes = b"\x38\x3c\x40\xb8"
  let first = payload_test_admit(Fp8E4M3Block128Parameter, [1L, 4L], bytes)
  let repeated = payload_test_admit(Fp8E4M3Block128Parameter, [1L, 4L], bytes)
  let other_content = payload_test_admit(
    Fp8E4M3Block128Parameter,
    [1L, 4L],
    bytes,
    fill='b',
  )
  let other_bytes = payload_test_admit(
    Fp8E4M3Block128Parameter,
    [1L, 4L],
    b"\x38\x3c\x40\x38",
  )
  assert_eq(first.schema_version(), 1)
  assert_eq(first.encoding(), Fp8E4M3Block128Parameter)
  assert_eq(first.block_shape().rows(), 128)
  assert_eq(first.block_shape().columns(), 128)
  assert_eq(first.logical_shape(), [1L, 4L])
  assert_eq(first.logical_element_count(), 4L)
  assert_eq(first.scalar_interpretation(), FiniteFp8E4M3Available)
  assert_eq(first.reference_decode_scalar(1L), 1.5)
  assert_eq(first.contract_digest(), repeated.contract_digest())
  assert_eq(first.payload_digest(), other_content.payload_digest())
  assert_true(first.contract_digest() != other_content.contract_digest())
  assert_true(first.payload_digest() != other_bytes.payload_digest())
  assert_true(first.contract_digest() != other_bytes.contract_digest())
}

///|
test "all exact block layouts validate their independent host byte counts" {
  let fp8 = payload_test_admit(
    Fp8E4M3Block128Parameter,
    [2L, 3L],
    Bytes::make(6, b'\x00'),
  )
  let block128 = payload_test_admit(
    Ue8M0Block128ScaleMetadata,
    [129L, 129L],
    Bytes::make(4, b'\x7f'),
  )
  let fp4 = payload_test_admit(
    PackedFp4E2M1Block32Parameter,
    [2L, 32L],
    Bytes::make(32, b'\x10'),
  )
  let block32 = payload_test_admit(
    Ue8M0Block32ScaleMetadata,
    [2L, 32L],
    Bytes::make(2, b'\x7f'),
  )
  assert_eq(fp8.payload_bytes().length(), 6)
  assert_eq(block128.payload_bytes().length(), 4)
  assert_eq(fp4.payload_bytes().length(), 32)
  assert_eq(fp4.fp4_codebook(), Some(CanonicalE2M1))
  assert_eq(fp8.fp4_codebook(), None)
  assert_eq(block32.payload_bytes().length(), 2)
}

///|
test "scale maps are exact while interpretation and dequantization stay unavailable" {
  let fp8 = payload_test_admit(
    Fp8E4M3Block128Parameter,
    [129L, 129L],
    Bytes::make(129 * 129, b'\x38'),
  )
  let fp8_scale = payload_test_admit(
    Ue8M0Block128ScaleMetadata,
    [129L, 129L],
    Bytes::make(4, b'\x7f'),
  )
  assert_eq(fp8.scale_index_for_scalar(fp8_scale, 0L), 0L)
  assert_eq(fp8.scale_index_for_scalar(fp8_scale, 127L), 0L)
  assert_eq(fp8.scale_index_for_scalar(fp8_scale, 128L), 1L)
  assert_eq(fp8.scale_index_for_scalar(fp8_scale, 129L * 128L), 2L)
  assert_eq(fp8.scale_index_for_scalar(fp8_scale, 129L * 129L - 1L), 3L)
  assert_eq(
    fp8_scale.scalar_interpretation(),
    Unavailable(Ue8M0ExponentBiasAndSpecialCodesUnproven),
  )
  try fp8_scale.reference_decode_scalar(0L) catch {
    DecodeUnavailable(Ue8M0ExponentBiasAndSpecialCodesUnproven) => ()
    _ => fail("unexpected UE8M0 decode rejection")
  } noraise {
    _ => fail("unproven UE8M0 byte was decoded")
  }
  try fp8.reference_dequantize_scalar(fp8_scale, 0L) catch {
    DecodeUnavailable(Ue8M0ExponentBiasAndSpecialCodesUnproven) => ()
    _ => fail("unexpected FP8 dequantization rejection")
  } noraise {
    _ => fail("FP8 payload dequantized through an unproven scale")
  }
}

///|
test "packed FP4 layout is admitted but codebook and nibble order stay unavailable" {
  let fp4 = payload_test_admit(
    PackedFp4E2M1Block32Parameter,
    [1L, 32L],
    Bytes::make(16, b'\x10'),
  )
  let scale = payload_test_admit(
    Ue8M0Block32ScaleMetadata,
    [1L, 32L],
    Bytes::make(1, b'\x7f'),
  )
  assert_eq(fp4.scale_index_for_scalar(scale, 31L), 0L)
  assert_eq(
    fp4.scalar_interpretation(),
    Unavailable(Fp4NibbleAndCodebookOrderingUnproven),
  )
  try fp4.reference_decode_scalar(0L) catch {
    DecodeUnavailable(Fp4NibbleAndCodebookOrderingUnproven) => ()
    _ => fail("unexpected FP4 decode rejection")
  } noraise {
    _ => fail("unproven FP4 nibble/codebook ordering was guessed")
  }
  try fp4.reference_dequantize_scalar(scale, 0L) catch {
    DecodeUnavailable(Fp4NibbleAndCodebookOrderingUnproven) => ()
    _ => fail("unexpected FP4 dequantization rejection")
  } noraise {
    _ => fail("unproven FP4 payload was dequantized")
  }
}

///|
test "hostile shape block byte and pairing mutations fail closed" {
  let limits = payload_test_limits()
  try
    admit(
      storage=PlainBFloat16,
      block_shape=try! DeepSeekV4QuantizedBlockShape::new(rows=128, columns=128),
      logical_shape=[1L, 1L],
      model_content_digest=payload_test_content(),
      payload_bytes=b"\x00",
      limits~,
    )
  catch {
    UnsupportedLogicalStorage(PlainBFloat16) => ()
    _ => fail("unexpected plain-storage rejection")
  } noraise {
    _ => fail("plain storage entered the quantized payload boundary")
  }
  try
    admit(
      storage=Fp8E4M3Block128Parameter,
      block_shape=try! DeepSeekV4QuantizedBlockShape::new(rows=1, columns=32),
      logical_shape=[1L, 1L],
      model_content_digest=payload_test_content(),
      payload_bytes=b"\x00",
      limits~,
    )
  catch {
    InvalidBlockShape => ()
    _ => fail("unexpected block-shape rejection")
  } noraise {
    _ => fail("wrong FP8 block geometry was admitted")
  }
  try payload_test_admit(Fp8E4M3Block128Parameter, [1L, 2L], b"\x00") catch {
    PayloadByteLengthMismatch(expected=2L, actual=1L) => ()
    _ => fail("unexpected payload byte-count rejection")
  } noraise {
    _ => fail("truncated FP8 payload was admitted")
  }
  try
    payload_test_admit(Fp8E4M3Block128Parameter, [1L, 2L], b"\x38\x7f")
  catch {
    NonFiniteFp8PayloadCode(element=1L, code=b'\x7f') => ()
    _ => fail("unexpected non-finite payload rejection")
  } noraise {
    _ => fail("non-finite E4M3 payload was admitted")
  }
  try
    payload_test_admit(
      PackedFp4E2M1Block32Parameter,
      [1L, 33L],
      Bytes::make(17, b'\x00'),
    )
  catch {
    LayoutUnavailable(Fp4OddColumnPaddingUnproven) => ()
    _ => fail("unexpected odd FP4 padding rejection")
  } noraise {
    _ => fail("unproven odd FP4 padding was guessed")
  }
  try
    payload_test_admit(
      Ue8M0Block32ScaleMetadata,
      [1L, 33L],
      Bytes::make(2, b'\x00'),
    )
  catch {
    LayoutUnavailable(Ue8M0Block32PartialPaddingUnproven) => ()
    _ => fail("unexpected partial block-32 rejection")
  } noraise {
    _ => fail("unproven block-32 padding was guessed")
  }
  let fp8 = payload_test_admit(Fp8E4M3Block128Parameter, [1L, 1L], b"\x38")
  let wrong_scale = payload_test_admit(
    Ue8M0Block128ScaleMetadata,
    [1L, 2L],
    b"\x7f",
  )
  try fp8.scale_index_for_scalar(wrong_scale, 0L) catch {
    ScalePairMismatch => ()
    _ => fail("unexpected scale pairing rejection")
  } noraise {
    _ => fail("shape-mismatched scale metadata was paired")
  }
  try fp8.reference_decode_scalar(1L) catch {
    ScalarIndexOutOfBounds(1L) => ()
    _ => fail("unexpected scalar-bound rejection")
  } noraise {
    _ => fail("out-of-bounds scalar was decoded")
  }
}

///|
test "hostile startup bounds and canonical ceiling reject before escape" {
  try
    DeepSeekV4QuantizedPayloadLimits::new(
      maximum_elements=0L,
      maximum_payload_bytes=1L,
      maximum_canonical_bytes=1,
    )
  catch {
    InvalidLimit => ()
    _ => fail("unexpected zero-bound rejection")
  } noraise {
    _ => fail("zero element bound was admitted")
  }
  try
    admit(
      storage=Fp8E4M3Block128Parameter,
      block_shape=payload_test_block(Fp8E4M3Block128Parameter),
      logical_shape=[1L, 1L],
      model_content_digest=payload_test_content(),
      payload_bytes=b"\x00",
      limits=payload_test_limits(maximum_canonical_bytes=1),
    )
  catch {
    CanonicalEncodingTooLarge => ()
    _ => fail("unexpected canonical ceiling rejection")
  } noraise {
    _ => fail("oversized canonical payload identity was admitted")
  }
}

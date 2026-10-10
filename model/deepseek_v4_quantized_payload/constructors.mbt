///|
pub fn DeepSeekV4QuantizedPayloadLimits::new(
  maximum_elements~ : Int64,
  maximum_payload_bytes~ : Int64,
  maximum_canonical_bytes~ : Int,
) -> Self raise DeepSeekV4QuantizedPayloadError {
  if maximum_elements <= 0L ||
    maximum_elements > 2147483647L ||
    maximum_payload_bytes <= 0L ||
    maximum_payload_bytes > 2147483647L ||
    maximum_canonical_bytes <= 0 ||
    maximum_canonical_bytes > 1048576 {
    raise InvalidLimit
  }
  { maximum_elements, maximum_payload_bytes, maximum_canonical_bytes, }
}

///|
pub fn DeepSeekV4QuantizedBlockShape::new(
  rows~ : Int,
  columns~ : Int,
) -> Self raise DeepSeekV4QuantizedPayloadError {
  if rows <= 0 || columns <= 0 || rows > 4096 || columns > 4096 {
    raise InvalidBlockShape
  }
  { rows, columns, }
}

///|
pub fn DeepSeekV4QuantizedBlockShape::rows(self : Self) -> Int {
  self.rows
}

///|
pub fn DeepSeekV4QuantizedBlockShape::columns(self : Self) -> Int {
  self.columns
}

///|
pub fn DeepSeekV4QuantizedPayloadContractDigest::as_hex(self : Self) -> String {
  self.hex
}

///|
pub fn DeepSeekV4QuantizedPayloadEncoding::from_logical_storage(
  storage : @deepseek_v4_numeric_plan.DeepSeekV4LogicalStorage,
) -> Self raise DeepSeekV4QuantizedPayloadError {
  match storage {
    Fp8E4M3Block128Parameter => Fp8E4M3Block128Parameter
    Ue8M0Block128ScaleMetadata => Ue8M0Block128ScaleMetadata
    PackedFp4E2M1Block32Parameter => PackedFp4E2M1Block32Parameter
    Ue8M0Block32ScaleMetadata => Ue8M0Block32ScaleMetadata
    _ => raise UnsupportedLogicalStorage(storage)
  }
}

///|
fn exact_block_shape(
  encoding : DeepSeekV4QuantizedPayloadEncoding,
) -> DeepSeekV4QuantizedBlockShape {
  match encoding {
    Fp8E4M3Block128Parameter | Ue8M0Block128ScaleMetadata =>
      { rows: 128, columns: 128, }
    PackedFp4E2M1Block32Parameter | Ue8M0Block32ScaleMetadata =>
      { rows: 1, columns: 32, }
  }
}

///|
fn checked_elements(
  shape : ArrayView[Int64],
  maximum : Int64,
) -> Int64 raise DeepSeekV4QuantizedPayloadError {
  if shape.length() != 2 {
    raise InvalidLogicalShape
  }
  let mut elements = 1L
  for dimension in shape {
    if dimension <= 0L || elements > maximum / dimension {
      raise ElementCountExceeded
    }
    elements *= dimension
  }
  elements
}

///|
fn expected_payload_bytes(
  encoding : DeepSeekV4QuantizedPayloadEncoding,
  shape : ArrayView[Int64],
  elements : Int64,
) -> Int64 raise DeepSeekV4QuantizedPayloadError {
  let rows = shape[0]
  let columns = shape[1]
  match encoding {
    Fp8E4M3Block128Parameter => elements
    Ue8M0Block128ScaleMetadata =>
      (rows + 127L) / 128L * ((columns + 127L) / 128L)
    PackedFp4E2M1Block32Parameter => {
      if columns % 2L != 0L {
        raise LayoutUnavailable(Fp4OddColumnPaddingUnproven)
      }
      rows * (columns / 2L)
    }
    Ue8M0Block32ScaleMetadata => {
      if columns % 32L != 0L {
        raise LayoutUnavailable(Ue8M0Block32PartialPaddingUnproven)
      }
      rows * (columns / 32L)
    }
  }
}

///|
/// Admit exact bounded host bytes without granting materialization authority.
pub fn admit(
  storage~ : @deepseek_v4_numeric_plan.DeepSeekV4LogicalStorage,
  block_shape~ : DeepSeekV4QuantizedBlockShape,
  logical_shape~ : ArrayView[Int64],
  model_content_digest~ : @spec.ContentDigest,
  payload_bytes~ : Bytes,
  limits~ : DeepSeekV4QuantizedPayloadLimits,
) -> DeepSeekV4QuantizedPayloadContract raise DeepSeekV4QuantizedPayloadError {
  let encoding = DeepSeekV4QuantizedPayloadEncoding::from_logical_storage(
    storage,
  )
  if block_shape != exact_block_shape(encoding) {
    raise InvalidBlockShape
  }
  let elements = checked_elements(logical_shape, limits.maximum_elements)
  let expected = expected_payload_bytes(encoding, logical_shape, elements)
  if expected > limits.maximum_payload_bytes {
    raise PayloadByteCountExceeded
  }
  let actual = payload_bytes.length().to_int64()
  if actual != expected {
    raise PayloadByteLengthMismatch(expected~, actual~)
  }
  if encoding is Fp8E4M3Block128Parameter {
    for index, code in payload_bytes {
      if code == b'\x7f' || code == b'\xff' {
        raise NonFiniteFp8PayloadCode(element=index.to_int64(), code~)
      }
    }
  }
  let copied_shape = ReadOnlyArray::from_array(logical_shape)
  let fp4_codebook = if encoding is PackedFp4E2M1Block32Parameter {
    Some(@deepseek_v4_numeric_plan.DeepSeekV4Fp4CodebookBinding::CanonicalE2M1)
  } else {
    None
  }
  let contract_digest = canonical_contract_digest(
    encoding,
    block_shape,
    copied_shape,
    model_content_digest,
    fp4_codebook,
    actual,
    limits.maximum_canonical_bytes,
  )
  {
    schema_version: 2,
    encoding,
    block_shape,
    logical_shape: copied_shape,
    logical_element_count: elements,
    payload_bytes,
    model_content_digest,
    fp4_codebook,
    contract_digest,
  }
}

///|
fn power_of_two(exponent : Int) -> Double {
  let mut value = 1.0
  if exponent >= 0 {
    for _ in 0..<exponent {
      value *= 2.0
    }
  } else {
    for _ in 0..<-exponent {
      value /= 2.0
    }
  }
  value
}

///|
/// Decode one proven finite E4M3FN bit pattern without a scale.
pub fn decode_finite_fp8_e4m3(
  code : Byte,
) -> Double raise DeepSeekV4QuantizedPayloadError {
  if code == b'\x7f' || code == b'\xff' {
    raise NonFiniteFp8Code(code)
  }
  let bits = code.to_int()
  let negative = (bits & 0x80) != 0
  let exponent = (bits >> 3) & 0x0f
  let mantissa = bits & 0x07
  let magnitude = if exponent == 0 {
    mantissa.to_double() * power_of_two(-9)
  } else {
    (1.0 + mantissa.to_double() / 8.0) * power_of_two(exponent - 7)
  }
  if negative {
    -magnitude
  } else {
    magnitude
  }
}

///|
pub fn DeepSeekV4QuantizedPayloadContract::scalar_interpretation(
  self : Self,
) -> DeepSeekV4ScalarInterpretationAvailability {
  match self.encoding {
    Fp8E4M3Block128Parameter => FiniteFp8E4M3Available
    Ue8M0Block128ScaleMetadata | Ue8M0Block32ScaleMetadata =>
      Unavailable(Ue8M0ExponentBiasAndSpecialCodesUnproven)
    PackedFp4E2M1Block32Parameter =>
      Unavailable(Fp4NibbleAndCodebookOrderingUnproven)
  }
}

///|
/// Decode one logical scalar only when its interpretation is proven.
pub fn DeepSeekV4QuantizedPayloadContract::reference_decode_scalar(
  self : Self,
  logical_index : Int64,
) -> Double raise DeepSeekV4QuantizedPayloadError {
  if logical_index < 0L || logical_index >= self.logical_element_count {
    raise ScalarIndexOutOfBounds(logical_index)
  }
  match self.encoding {
    Fp8E4M3Block128Parameter =>
      decode_finite_fp8_e4m3(self.payload_bytes[logical_index.to_int()])
    Ue8M0Block128ScaleMetadata | Ue8M0Block32ScaleMetadata =>
      raise DecodeUnavailable(Ue8M0ExponentBiasAndSpecialCodesUnproven)
    PackedFp4E2M1Block32Parameter =>
      raise DecodeUnavailable(Fp4NibbleAndCodebookOrderingUnproven)
  }
}

///|
fn same_logical_shape(
  left : ArrayView[Int64],
  right : ArrayView[Int64],
) -> Bool {
  if left.length() != right.length() {
    return false
  }
  for index, dimension in left {
    if right[index] != dimension {
      return false
    }
  }
  true
}

///|
fn exact_scale_pair(
  parameter : DeepSeekV4QuantizedPayloadContract,
  scale : DeepSeekV4QuantizedPayloadContract,
) -> Bool {
  if parameter.model_content_digest != scale.model_content_digest ||
    !same_logical_shape(parameter.logical_shape, scale.logical_shape) {
    return false
  }
  match (parameter.encoding, scale.encoding) {
    (Fp8E4M3Block128Parameter, Ue8M0Block128ScaleMetadata)
    | (PackedFp4E2M1Block32Parameter, Ue8M0Block32ScaleMetadata) => true
    _ => false
  }
}

///|
fn scale_index_for(
  parameter : DeepSeekV4QuantizedPayloadContract,
  logical_index : Int64,
) -> Int64 {
  let columns = parameter.logical_shape[1]
  let row = logical_index / columns
  let column = logical_index % columns
  match parameter.encoding {
    Fp8E4M3Block128Parameter => {
      let scale_columns = (columns + 127L) / 128L
      row / 128L * scale_columns + column / 128L
    }
    PackedFp4E2M1Block32Parameter => row * (columns / 32L) + column / 32L
    _ => abort("scale index requested for scale metadata")
  }
}

///|
/// Return the exact scale-cell mapping without interpreting scale bytes.
pub fn DeepSeekV4QuantizedPayloadContract::scale_index_for_scalar(
  self : Self,
  scale : DeepSeekV4QuantizedPayloadContract,
  logical_index : Int64,
) -> Int64 raise DeepSeekV4QuantizedPayloadError {
  if logical_index < 0L || logical_index >= self.logical_element_count {
    raise ScalarIndexOutOfBounds(logical_index)
  }
  if !exact_scale_pair(self, scale) {
    raise ScalePairMismatch
  }
  scale_index_for(self, logical_index)
}

///|
/// Fail closed until UE8M0 and, for FP4, nibble/codebook semantics are proven.
pub fn DeepSeekV4QuantizedPayloadContract::reference_dequantize_scalar(
  self : Self,
  scale : DeepSeekV4QuantizedPayloadContract,
  logical_index : Int64,
) -> Double raise DeepSeekV4QuantizedPayloadError {
  ignore(self.scale_index_for_scalar(scale, logical_index))
  match self.encoding {
    Fp8E4M3Block128Parameter => {
      ignore(self.reference_decode_scalar(logical_index))
      raise DecodeUnavailable(Ue8M0ExponentBiasAndSpecialCodesUnproven)
    }
    PackedFp4E2M1Block32Parameter =>
      raise DecodeUnavailable(Fp4NibbleAndCodebookOrderingUnproven)
    _ => raise ScalePairMismatch
  }
}

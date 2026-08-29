///|
fn fused_qualification_bf16_bits(value : Float) -> Int {
  let bits = value.reinterpret_as_uint()
  ((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16).reinterpret_as_int()
}

///|
fn fused_qualification_bf16_float(bits : Int) -> Float {
  Float::reinterpret_from_uint((bits.reinterpret_as_uint() & 0xffffU) << 16)
}

///|
pub fn qualification_bf16_bytes(values : ArrayView[Int]) -> Bytes {
  let output = FixedArray::make(values.length() * 2, b'\x00')
  for index, value in values {
    output[index * 2] = value.to_byte()
    output[index * 2 + 1] = (value >> 8).to_byte()
  }
  Bytes::from_array(output[:])
}

///|
fn fused_qualification_error_ppb(value : Float) -> Int64? {
  if value.is_nan() || value.is_inf() || value < 0.0F {
    return None
  }
  let scaled = value.to_double() * 1000000000.0
  if scaled.is_nan() || scaled.is_inf() || scaled > 9223372036854775807.0 {
    None
  } else {
    Some(scaled.ceil().to_int64())
  }
}

///|
fn fused_qualification_compare_bf16(
  actual : ArrayView[Int],
  expected : ArrayView[Int],
  maximum_absolute_error_ppb : Int64,
  maximum_relative_error_ppb : Int64,
) -> FusedQualificationComparisonSummary raise FusedQualificationError {
  if actual.length() != expected.length() ||
    maximum_absolute_error_ppb < 0L ||
    maximum_relative_error_ppb < 0L {
    raise InvalidRefereeInput
  }
  let mut observed_absolute = 0.0F
  let mut observed_absolute_ppb = 0L
  let mut observed_relative_ppb = 0L
  for index in 0..<actual.length() {
    let actual_value = fused_qualification_bf16_float(actual[index])
    let expected_value = fused_qualification_bf16_float(expected[index])
    if actual_value.is_nan() ||
      actual_value.is_inf() ||
      expected_value.is_nan() ||
      expected_value.is_inf() {
      raise RefereeMismatch(index~)
    }
    let difference = (actual_value - expected_value).abs()
    let absolute_ppb = match fused_qualification_error_ppb(difference) {
      Some(value) => value
      None => raise RefereeMismatch(index~)
    }
    let relative_ppb = if difference == 0.0F {
      0L
    } else {
      let denominator = expected_value.abs()
      if denominator == 0.0F {
        raise RefereeMismatch(index~)
      } else {
        match fused_qualification_error_ppb(difference / denominator) {
          Some(value) => value
          None => raise RefereeMismatch(index~)
        }
      }
    }
    if absolute_ppb > maximum_absolute_error_ppb ||
      relative_ppb > maximum_relative_error_ppb {
      raise RefereeMismatch(index~)
    }
    if difference > observed_absolute {
      observed_absolute = difference
    }
    if absolute_ppb > observed_absolute_ppb {
      observed_absolute_ppb = absolute_ppb
    }
    if relative_ppb > observed_relative_ppb {
      observed_relative_ppb = relative_ppb
    }
  }
  {
    maximum_absolute_error_ppb: observed_absolute_ppb,
    maximum_relative_error_ppb: observed_relative_ppb,
    maximum_absolute_error: observed_absolute,
  }
}

///|
/// Applies both fixed QKV numerical limits and returns their measured PPB
/// maxima. Relative error is computed per cell against the exact scalar BF16
/// expectation; a nonzero result against exact zero fails closed.
pub fn compare_qkv_bf16_output(
  actual : ArrayView[Int],
  expected : ArrayView[Int],
) -> FusedQualificationComparisonSummary raise FusedQualificationError {
  fused_qualification_compare_bf16(actual, expected, 250000L, 500000L)
}

///|
/// Applies both fixed residual/RMSNorm numerical limits and returns their
/// measured PPB maxima.
pub fn compare_residual_norm_bf16_output(
  actual : ArrayView[Int],
  expected : ArrayView[Int],
) -> FusedQualificationComparisonSummary raise FusedQualificationError {
  fused_qualification_compare_bf16(actual, expected, 100000L, 250000L)
}

///|
/// Applies the fused QKV/RoPE/KV-write candidate's fixed absolute numerical
/// contract. Callers cannot widen the qualification tolerance.
pub fn qualify_qkv_bf16_output(
  actual : ArrayView[Int],
  expected : ArrayView[Int],
) -> Float raise FusedQualificationError {
  compare_qkv_bf16_output(actual, expected).maximum_absolute_error()
}

///|
/// Applies the fused residual/RMSNorm candidate's fixed absolute numerical
/// contract. Callers cannot widen the qualification tolerance.
pub fn qualify_residual_norm_bf16_output(
  actual : ArrayView[Int],
  expected : ArrayView[Int],
) -> Float raise FusedQualificationError {
  compare_residual_norm_bf16_output(actual, expected).maximum_absolute_error()
}

///|
/// IEEE BF16 round-to-nearest-even, represented as FP32. Overflow becomes
/// infinity; NaNs remain NaNs. This is not a finite-saturating quantizer.
pub fn round_bf16(value : Float) -> Float {
  let bits = value.reinterpret_as_uint()
  if (bits & 0x7fffffffU) > 0x7f800000U {
    Float::reinterpret_from_uint((bits & 0xffff0000U) | 0x00400000U)
  } else {
    Float::reinterpret_from_uint(
      (bits + 0x7fffU + ((bits >> 16) & 1U)) & 0xffff0000U,
    )
  }
}

///|
fn exp32(value : Float) -> Float {
  @math.expf(value)
}

///|
fn sigmoid_bf16(value : Float) -> Float {
  round_bf16(1.0F / (1.0F + exp32(-round_bf16(value))))
}

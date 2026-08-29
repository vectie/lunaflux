///|
fn decode_f32_bits_le(source : FixedArray[Byte], offset : Int) -> UInt {
  source[offset].to_uint() |
  (source[offset + 1].to_uint() << 8) |
  (source[offset + 2].to_uint() << 16) |
  (source[offset + 3].to_uint() << 24)
}

///|
fn fp8_scale_cell_is_finite_positive(bits : UInt) -> Bool {
  let exponent = bits & 0x7f800000U
  let sign = bits & 0x80000000U
  let magnitude = bits & 0x7fffffffU
  sign == 0U && exponent != 0x7f800000U && magnitude != 0U
}

///|
fn validate_fp8_scale_scratch(
  scratch : FixedArray[Byte],
  byte_count : Int64,
) -> Bool {
  if (byte_count != 4L && byte_count != 8L) ||
    byte_count > scratch.length().to_int64() {
    return false
  }
  for offset = 0; offset < byte_count.to_int(); offset = offset + 4 {
    if !fp8_scale_cell_is_finite_positive(decode_f32_bits_le(scratch, offset)) {
      return false
    }
  }
  true
}

///|
/// Allocation-free publication seam shared by the native launch path and
/// hostile fake-device tests. Output authority requires all three independent
/// observations; a successful launch alone never publishes output.
fn admit_fp8_execution_evidence(
  launch_complete : Bool,
  readback_complete : Bool,
  scratch : FixedArray[Byte],
  byte_count : Int64,
) -> Bool {
  launch_complete &&
  readback_complete &&
  validate_fp8_scale_scratch(scratch, byte_count)
}

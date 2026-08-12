///|
/// Strictly decodes a little-endian BF16 payload into a finite rank-2 tensor.
///
/// Each two-byte code unit is interpreted as the high 16 bits of one IEEE-754
/// binary32 value. The byte count must match the declared shape exactly.
pub fn decode_bfloat16_le(
  bytes : BytesView,
  rows~ : Int,
  columns~ : Int,
  limits~ : ReferenceLimits,
) -> Tensor raise ReferenceKernelError {
  let elements = checked_element_count(rows, columns, limits)
  let expected_bytes = checked_product(elements, 2)
  if bytes.length() != expected_bytes {
    raise InvalidShape(Bfloat16ByteCount)
  }
  let values : Array[Float] = []
  for index in 0..<elements {
    let offset = index * 2
    let low = bytes[offset].to_uint()
    let high = bytes[offset + 1].to_uint()
    let bits = ((high << 8) | low) << 16
    let value = Float::reinterpret_from_uint(bits)
    if value.is_nan() || value.is_inf() {
      raise NonFinite(Bfloat16Value)
    }
    values.push(value)
  }
  { rows, columns, values: ReadOnlyArray::from_array(values) }
}

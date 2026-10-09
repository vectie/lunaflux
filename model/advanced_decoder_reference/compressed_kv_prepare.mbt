///|
/// Materializes one borrowed shared-K/V row using the exact official segment
/// layout. Prefill appends the complete compressed prefix after all ordinary
/// sequence rows. Decode preserves the circular window's physical slot order
/// and appends the completed compressed-cache prefix. Values are copied
/// bit-for-bit at the BF16 boundary; this oracle uses `Float` only as a bounded
/// host carrier and performs no arithmetic.
pub fn non_overlapping_compressed_kv_prepare_v1(
  ordinary_key_value : ArrayView[Float],
  compressed_key_value : ArrayView[Float],
  window_size~ : Int,
  compression_ratio~ : Int,
  head_dimension~ : Int,
  sequence_length~ : Int,
  maximum_position_count~ : Int,
  decode_mode~ : Bool,
  limits~ : ReferenceLimits,
) -> ReadOnlyArray[Float] raise AdvancedDecoderReferenceError {
  if window_size <= 0 ||
    compression_ratio <= 1 ||
    head_dimension <= 0 ||
    sequence_length <= 0 ||
    maximum_position_count <= 0 ||
    window_size > maximum_position_count ||
    compression_ratio > maximum_position_count ||
    sequence_length > maximum_position_count {
    raise InvalidDimension
  }
  let ordinary_slots = if decode_mode { window_size } else { sequence_length }
  let compressed_slots = sequence_length / compression_ratio
  if ordinary_slots.to_int64() * head_dimension.to_int64() !=
    ordinary_key_value.length().to_int64() ||
    compressed_slots.to_int64() * head_dimension.to_int64() !=
    compressed_key_value.length().to_int64() ||
    ordinary_slots + compressed_slots > limits.max_width() ||
    ordinary_key_value.length() > limits.max_values() ||
    compressed_key_value.length() > limits.max_values() ||
    ordinary_key_value.length() >
    limits.max_values() - compressed_key_value.length() {
    raise InvalidDimension
  }
  validate_values(ordinary_key_value, limits)
  validate_values(compressed_key_value, limits)
  let prepared : Array[Float] = []
  for value in ordinary_key_value {
    prepared.push(value)
  }
  for value in compressed_key_value {
    prepared.push(value)
  }
  ReadOnlyArray::from_array(prepared)
}

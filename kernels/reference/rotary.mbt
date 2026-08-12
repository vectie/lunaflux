///|
/// Applies split-half Llama rotary embedding to the Q and K regions of QKV.
///
/// Rows are consecutive token positions beginning at `position_offset`. The V
/// region is copied unchanged. `head_dimension` must be even.
pub fn rotary_embedding(
  qkv : Activation,
  query_heads~ : Int,
  kv_heads~ : Int,
  head_dimension~ : Int,
  position_offset~ : Int,
  rope_theta~ : Float,
  limits~ : ReferenceLimits,
) -> Activation raise ReferenceKernelError {
  if query_heads <= 0 ||
    kv_heads <= 0 ||
    head_dimension <= 0 ||
    head_dimension % 2 != 0 {
    raise InvalidShape(RotaryLayout)
  }
  if position_offset < 0 ||
    (qkv.rows > 0 && position_offset > 2147483647 - (qkv.rows - 1)) {
    raise InvalidPosition
  }
  if rope_theta.is_nan() || rope_theta.is_inf() {
    raise NonFinite(RopeTheta)
  }
  if rope_theta <= 0.0F {
    raise InvalidScalar(RopeTheta)
  }
  let query_width = checked_product(query_heads, head_dimension)
  let kv_width = checked_product(kv_heads, head_dimension)
  let expected_width = checked_sum(query_width, checked_product(kv_width, 2))
  if qkv.columns != expected_width {
    raise InvalidShape(RotaryLayout)
  }
  let elements = checked_element_count(qkv.rows, qkv.columns, limits)
  let output = Array::make(elements, 0.0F)
  for index in 0..<elements {
    output[index] = qkv.values[index]
  }
  let half = head_dimension / 2
  let total_rotary_heads = checked_sum(query_heads, kv_heads)
  for row in 0..<qkv.rows {
    let position = Float::from_int(position_offset + row)
    let row_offset = row * qkv.columns
    for head in 0..<total_rotary_heads {
      let head_offset = row_offset + head * head_dimension
      for pair in 0..<half {
        let exponent = Float::from_int(pair * 2) /
          Float::from_int(head_dimension)
        let angle = position / @math.powf(rope_theta, exponent)
        let cosine = @math.cosf(angle)
        let sine = @math.sinf(angle)
        let first = qkv.values[head_offset + pair]
        let second = qkv.values[head_offset + pair + half]
        output[head_offset + pair] = first * cosine - second * sine
        output[head_offset + pair + half] = second * cosine + first * sine
      }
    }
  }
  activation_from_owned(qkv.rows, qkv.columns, output, limits)
}

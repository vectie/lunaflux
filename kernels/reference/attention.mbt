///|
/// Computes full-recomputation causal grouped-query attention.
///
/// Input columns are concatenated Q, K, and V regions. Query heads are mapped
/// to KV heads in contiguous equal-sized groups. Output columns contain query
/// heads in head-major order.
pub fn causal_gqa_attention(
  qkv : Activation,
  query_heads~ : Int,
  kv_heads~ : Int,
  head_dimension~ : Int,
  limits~ : ReferenceLimits,
) -> Activation raise ReferenceKernelError {
  if query_heads <= 0 ||
    kv_heads <= 0 ||
    head_dimension <= 0 ||
    kv_heads > query_heads ||
    query_heads % kv_heads != 0 {
    raise InvalidShape(AttentionHeadRelationship)
  }
  let query_width = checked_product(query_heads, head_dimension)
  let kv_width = checked_product(kv_heads, head_dimension)
  let expected_width = checked_sum(query_width, checked_product(kv_width, 2))
  if qkv.columns != expected_width {
    raise InvalidShape(AttentionLayout)
  }
  let output_elements = checked_element_count(qkv.rows, query_width, limits)
  let output = Array::make(output_elements, 0.0F)
  // The positive query width means the checked output element count also
  // bounds this one-score-per-row scratch array.
  let scores = Array::make(qkv.rows, 0.0F)
  let group_size = query_heads / kv_heads
  let scale = 1.0F / Float::from_int(head_dimension).sqrt()
  let key_region = query_width
  let value_region = query_width + kv_width
  for query_row in 0..<qkv.rows {
    let query_row_offset = query_row * qkv.columns
    let output_row_offset = query_row * query_width
    for query_head in 0..<query_heads {
      let kv_head = query_head / group_size
      let query_offset = query_row_offset + query_head * head_dimension
      for key_row in 0..<=query_row {
        let key_offset = key_row * qkv.columns +
          key_region +
          kv_head * head_dimension
        let mut dot = 0.0F
        for component in 0..<head_dimension {
          dot += qkv.values[query_offset + component] *
            qkv.values[key_offset + component]
        }
        let score = dot * scale
        if score.is_nan() || score.is_inf() {
          raise NonFinite(ComputedValue)
        }
        scores[key_row] = score
      }
      let mut maximum = scores[0]
      for key_row in 1..<=query_row {
        maximum = maximum.max(scores[key_row])
      }
      let mut denominator = 0.0F
      for key_row in 0..<=query_row {
        let probability = @math.expf(scores[key_row] - maximum)
        scores[key_row] = probability
        denominator += probability
      }
      if denominator.is_nan() || denominator.is_inf() || denominator <= 0.0F {
        raise NonFinite(ComputedValue)
      }
      let output_offset = output_row_offset + query_head * head_dimension
      for component in 0..<head_dimension {
        let mut accumulator = 0.0F
        for key_row in 0..<=query_row {
          let value_offset = key_row * qkv.columns +
            value_region +
            kv_head * head_dimension +
            component
          accumulator += scores[key_row] /
            denominator *
            qkv.values[value_offset]
        }
        output[output_offset + component] = accumulator
      }
    }
  }
  activation_from_owned(qkv.rows, query_width, output, limits)
}

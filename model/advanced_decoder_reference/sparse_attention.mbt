///|
/// Bounded scalar oracle for selected-index attention over a shared K/V
/// buffer. `query` is `[head,component]`, `key_value` is
/// `[cache_slot,component]`, `sink` is `[head]`, and `selected_indices` is one
/// already-joined window+compressed row. `-1` entries are padding. Duplicate
/// valid indices retain their multiplicity, matching the gather contract.
///
/// The F32 sink participates only in the stable-softmax denominator. It never
/// contributes a value-vector numerator. This oracle deliberately excludes
/// index construction, compressed-KV preparation, inverse RoPE, and output
/// projection.
pub fn selected_sparse_attention(
  query : ArrayView[Float],
  key_value : ArrayView[Float],
  sink : ArrayView[Float],
  selected_indices : ArrayView[Int],
  head_count~ : Int,
  head_dimension~ : Int,
  limits~ : ReferenceLimits,
) -> ReadOnlyArray[Float] raise AdvancedDecoderReferenceError {
  validate_shape(head_count, head_dimension, limits)
  if query.length() != head_count * head_dimension ||
    sink.length() != head_count ||
    key_value.is_empty() ||
    key_value.length() % head_dimension != 0 ||
    selected_indices.is_empty() ||
    selected_indices.length() > limits.max_width ||
    Int64::from_int(query.length()) +
    Int64::from_int(key_value.length()) +
    Int64::from_int(sink.length()) >
    Int64::from_int(limits.max_values) {
    raise InvalidDimension
  }
  validate_values(query, limits)
  validate_values(key_value, limits)
  validate_values(sink, limits)
  let cache_slots = key_value.length() / head_dimension
  for index in selected_indices {
    if index < -1 || index >= cache_slots {
      raise KeyIndexOutOfRange(index)
    }
  }
  let scale = 1.0F / Float::from_int(head_dimension).sqrt()
  let output = Array::make(head_count * head_dimension, 0.0F)
  for head in 0..<head_count {
    let mut maximum = sink[head]
    for selected in selected_indices {
      if selected >= 0 {
        let mut dot = 0.0F
        for component in 0..<head_dimension {
          dot += query[head * head_dimension + component] *
            key_value[selected * head_dimension + component]
        }
        let score = dot * scale
        if !finite(score) {
          raise NonFiniteParameter
        }
        if score > maximum {
          maximum = score
        }
      }
    }
    let mut denominator = @math.expf(sink[head] - maximum)
    for selected in selected_indices {
      if selected >= 0 {
        let mut dot = 0.0F
        for component in 0..<head_dimension {
          dot += query[head * head_dimension + component] *
            key_value[selected * head_dimension + component]
        }
        denominator += @math.expf(dot * scale - maximum)
      }
    }
    if !finite(denominator) || denominator <= 0.0F {
      raise ZeroNormalization
    }
    for component in 0..<head_dimension {
      let mut numerator = 0.0F
      for selected in selected_indices {
        if selected >= 0 {
          let mut dot = 0.0F
          for inner in 0..<head_dimension {
            dot += query[head * head_dimension + inner] *
              key_value[selected * head_dimension + inner]
          }
          numerator += @math.expf(dot * scale - maximum) *
            key_value[selected * head_dimension + component]
        }
      }
      let value = numerator / denominator
      if !finite(value) {
        raise NonFiniteParameter
      }
      output[head * head_dimension + component] = value
    }
  }
  ReadOnlyArray::from_array(output)
}

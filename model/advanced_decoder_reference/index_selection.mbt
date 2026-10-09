///|
/// Computes the official learned compressed-index score shape: each key is
/// dotted with every query head, negative head scores are clamped to zero,
/// then learned per-head weights combine them. Ties prefer lower key indices.
pub fn learned_index_top_k(
  query_heads : ArrayView[Float],
  keys : ArrayView[Float],
  head_weights : ArrayView[Float],
  head_count~ : Int,
  head_dimension~ : Int,
  top_k~ : Int,
  limits~ : ReferenceLimits,
) -> ReferenceTopK raise AdvancedDecoderReferenceError {
  validate_shape(head_count, head_dimension, limits)
  if query_heads.length() != head_count * head_dimension ||
    head_weights.length() != head_count ||
    keys.is_empty() ||
    keys.length() % head_dimension != 0 ||
    !combined_values_within_limit(
      query_heads.length(),
      keys.length(),
      head_weights.length(),
      limits,
    ) {
    raise InvalidDimension
  }
  let key_count = keys.length() / head_dimension
  if top_k <= 0 || top_k > key_count {
    raise InvalidTopK
  }
  validate_values(query_heads, limits)
  validate_values(keys, limits)
  validate_values(head_weights, limits)
  let scores = Array::make(key_count, 0.0F)
  for key in 0..<key_count {
    for head in 0..<head_count {
      let mut dot = 0.0F
      for column in 0..<head_dimension {
        dot += query_heads[head * head_dimension + column] *
          keys[key * head_dimension + column]
      }
      if dot > 0.0F {
        scores[key] += dot * head_weights[head]
      }
    }
    if !finite(scores[key]) {
      raise NonFiniteParameter
    }
  }
  let indices = stable_top_k_indices(scores, top_k)
  let selected_scores : Array[Float] = []
  for index in indices {
    selected_scores.push(scores[index])
  }
  {
    indices: ReadOnlyArray::from_array(indices),
    scores: ReadOnlyArray::from_array(selected_scores),
  }
}

///|
/// Returns exactly the causal compressed-cache slots available to one absolute
/// query position, matching `floor((query_position + 1) / ratio)` semantics.
pub fn deterministic_compressed_indices(
  compression_ratio~ : Int,
  query_position~ : Int,
  cache_offset~ : Int,
  limits~ : ReferenceLimits,
) -> ReadOnlyArray[Int] raise AdvancedDecoderReferenceError {
  if compression_ratio <= 1 || query_position < 0 || cache_offset < 0 {
    raise InvalidCompression
  }
  let completed_tail = if query_position % compression_ratio ==
    compression_ratio - 1 {
    1
  } else {
    0
  }
  let count = query_position / compression_ratio + completed_tail
  if count > limits.max_values {
    raise InvalidDimension
  }
  let indices : Array[Int] = []
  for index in 0..<count {
    let absolute = Int64::from_int(cache_offset) + Int64::from_int(index)
    if absolute > 2147483647L {
      raise InvalidCompression
    }
    indices.push(absolute.to_int())
  }
  ReadOnlyArray::from_array(indices)
}

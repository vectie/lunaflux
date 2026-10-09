///|
fn oracle_validate_selected_indices(
  selected : ArrayView[Int],
  token_count : Int,
  head_count : Int,
  top_k : Int,
) -> Unit raise GlmHybridReferenceError {
  if selected.length() != token_count * head_count * top_k {
    raise InvalidShape
  }
  for token in 0..<token_count {
    for head in 0..<head_count {
      let base = (token * head_count + head) * top_k
      let mut visible = 0
      for rank in 0..<top_k {
        let index = selected[base + rank]
        if index == -1 {
          continue
        }
        if index < 0 || index > token || index >= token_count {
          raise IndexOutOfBounds
        }
        visible += 1
        for previous in 0..<rank {
          if index >= 0 && selected[base + previous] == index {
            raise DuplicateIndex
          }
        }
      }
      if visible == 0 {
        raise IndexOutOfBounds
      }
    }
  }
}

///|
fn oracle_sparse_attention(
  query : ArrayView[Double],
  key : ArrayView[Double],
  value : ArrayView[Double],
  selected : ArrayView[Int],
  token_count : Int,
  head_count : Int,
  head_dimension : Int,
  top_k : Int,
  limits : GlmTinyOracleLimits,
) -> Array[Double] raise GlmHybridReferenceError {
  let width = head_count * head_dimension
  if query.length() != token_count * width ||
    key.length() != query.length() ||
    value.length() != query.length() {
    raise InvalidShape
  }
  oracle_require_finite(query, limits)
  oracle_require_finite(key, limits)
  oracle_require_finite(value, limits)
  oracle_validate_selected_indices(selected, token_count, head_count, top_k)
  let output = Array::make(token_count * width, 0.0)
  let scores = Array::make(top_k, 0.0)
  let scale = 1.0 / head_dimension.to_double().sqrt()
  for token in 0..<token_count {
    for head in 0..<head_count {
      let selected_base = (token * head_count + head) * top_k
      let query_base = token * width + head * head_dimension
      let mut maximum = -@double.infinity
      for rank in 0..<top_k {
        let source = selected[selected_base + rank]
        if source == -1 {
          scores[rank] = -@double.infinity
          continue
        }
        let key_base = source * width + head * head_dimension
        let mut dot = 0.0
        for component in 0..<head_dimension {
          dot += query[query_base + component] * key[key_base + component]
        }
        let score = dot * scale
        if !finite(score) {
          raise NonFiniteInput
        }
        scores[rank] = score
        if score > maximum {
          maximum = score
        }
      }
      let mut denominator = 0.0
      for rank in 0..<top_k {
        scores[rank] = if scores[rank] == -@double.infinity {
          0.0
        } else {
          @math.exp(scores[rank] - maximum)
        }
        denominator += scores[rank]
      }
      if !finite(denominator) || denominator <= 0.0 {
        raise InvalidProbability
      }
      for rank in 0..<top_k {
        let source = selected[selected_base + rank]
        if source == -1 {
          continue
        }
        let source_base = source * width + head * head_dimension
        let probability = scores[rank] / denominator
        for component in 0..<head_dimension {
          output[query_base + component] += probability *
            value[source_base + component]
        }
      }
    }
  }
  oracle_require_finite(output, limits)
  output
}

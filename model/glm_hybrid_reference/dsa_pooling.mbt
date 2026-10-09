///|
/// Rebuilds GLM-5.3-Flash compressed index pools for one tiny batch. Pooling
/// begins at the first valid token. Only complete all-valid pools are emitted;
/// an incomplete tail is handled separately by the caller's selected-index
/// contract, exactly as in the model reference.
pub fn pool_dsa_index_keys(
  keys : ArrayView[Double],
  gate_scores : ArrayView[Double],
  valid_tokens : ArrayView[Bool],
  sequence_length~ : Int,
  head_dimension~ : Int,
  pool_size~ : Int,
  absolute_pool_encoding~ : ArrayView[Double],
  limits~ : GlmHybridReferenceLimits,
) -> GlmDsaPooledKeys raise GlmHybridReferenceError {
  limits.check_sequence(sequence_length)
  limits.check_heads(1, head_dimension)
  if pool_size <= 0 || pool_size > sequence_length {
    raise InvalidShape
  }
  if keys.length() != sequence_length * head_dimension ||
    gate_scores.length() != keys.length() ||
    valid_tokens.length() != sequence_length ||
    absolute_pool_encoding.length() != pool_size * head_dimension {
    raise InvalidShape
  }
  require_finite(keys)
  require_finite(gate_scores)
  require_finite(absolute_pool_encoding)
  let mut first_valid = sequence_length
  for index, valid in valid_tokens {
    if valid {
      first_valid = index
      break
    }
  }
  let available = sequence_length - first_valid
  let complete_pools = available / pool_size
  let pooled_keys = []
  let raw_indices = []
  let pool_valid = []
  for pool in 0..<complete_pools {
    let raw_start = first_valid + pool * pool_size
    let mut valid = true
    for offset in 0..<pool_size {
      if !valid_tokens[raw_start + offset] {
        valid = false
      }
    }
    if valid {
      for offset in 0..<pool_size {
        raw_indices.push(raw_start + offset)
      }
      for component in 0..<head_dimension {
        let mut maximum = @double.neg_infinity
        for offset in 0..<pool_size {
          let raw = raw_start + offset
          let logit = gate_scores[raw * head_dimension + component] +
            absolute_pool_encoding[offset * head_dimension + component]
          if logit > maximum {
            maximum = logit
          }
        }
        let mut denominator = 0.0
        let mut numerator = 0.0
        for offset in 0..<pool_size {
          let raw = raw_start + offset
          let probability = @math.exp(
            gate_scores[raw * head_dimension + component] +
            absolute_pool_encoding[offset * head_dimension + component] -
            maximum,
          )
          denominator += probability
          numerator += probability * keys[raw * head_dimension + component]
        }
        pooled_keys.push(numerator / denominator)
      }
      pool_valid.push(true)
    }
  }
  {
    pool_count: pool_valid.length(),
    pool_size,
    head_dimension,
    keys: ReadOnlyArray::from_array(pooled_keys),
    raw_indices: ReadOnlyArray::from_array(raw_indices),
    valid: ReadOnlyArray::from_array(pool_valid),
  }
}

///|
/// Computes official GLM DSA pooled candidate scores after projection: scaled
/// q·k, ReLU, learned per-query head weights, and head reduction. Selection is
/// intentionally delegated to `advanced_decoder_reference`'s stable top-k.
pub fn score_dsa_pools(
  queries : ArrayView[Double],
  query_head_weights : ArrayView[Double],
  pooled : GlmDsaPooledKeys,
  candidate_visible : ArrayView[Bool],
  query_count~ : Int,
  head_count~ : Int,
  limits~ : GlmHybridReferenceLimits,
) -> ReadOnlyArray[Double] raise GlmHybridReferenceError {
  limits.check_sequence(query_count)
  limits.check_heads(head_count, pooled.head_dimension)
  let dimension = pooled.head_dimension
  if queries.length() != query_count * head_count * dimension ||
    query_head_weights.length() != query_count * head_count ||
    candidate_visible.length() != query_count * pooled.pool_count {
    raise InvalidShape
  }
  require_finite(queries)
  require_finite(query_head_weights)
  let scores = []
  let scale = 1.0 / dimension.to_double().sqrt()
  let head_scale = 1.0 / head_count.to_double().sqrt()
  for query in 0..<query_count {
    for pool in 0..<pooled.pool_count {
      if !candidate_visible[query * pooled.pool_count + pool] {
        scores.push(@double.neg_infinity)
      } else {
        let mut reduced = 0.0
        for head in 0..<head_count {
          let mut dot = 0.0
          for component in 0..<dimension {
            dot += queries[(query * head_count + head) * dimension + component] *
              pooled.keys[pool * dimension + component]
          }
          let activated = if dot * scale > 0.0 { dot * scale } else { 0.0 }
          reduced += query_head_weights[query * head_count + head] *
            head_scale *
            activated
        }
        scores.push(reduced)
      }
    }
  }
  ReadOnlyArray::from_array(scores)
}

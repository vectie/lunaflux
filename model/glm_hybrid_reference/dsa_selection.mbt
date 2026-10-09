///|
/// Selects visible compressed pools with the shared stable learned-index
/// top-k primitive, expands those pools to raw token indices, then records the
/// GLM IndexShare state. Official DSA's `1/sqrt(D)` and `1/sqrt(H)` factors are
/// positive global score scales, so omitting them here cannot change top-k.
pub fn select_dsa_pools(
  query_heads : ArrayView[Double],
  head_weights : ArrayView[Double],
  pooled : GlmDsaPooledKeys,
  candidate_pools : ArrayView[Int],
  head_count~ : Int,
  top_k_pools~ : Int,
  source_layer~ : Int,
  key_count~ : Int,
  limits~ : GlmHybridReferenceLimits,
  advanced_limits~ : @advanced_decoder_reference.ReferenceLimits,
) -> GlmDsaTopKState raise GlmHybridReferenceError {
  limits.check_sequence(key_count)
  limits.check_heads(head_count, pooled.head_dimension)
  if query_heads.length() != head_count * pooled.head_dimension ||
    head_weights.length() != head_count ||
    candidate_pools.is_empty() ||
    top_k_pools <= 0 ||
    top_k_pools > candidate_pools.length() {
    raise InvalidShape
  }
  require_finite(query_heads)
  require_finite(head_weights)
  let seen : Map[Int, Unit] = Map([])
  let candidate_keys : Array[Float] = []
  for pool in candidate_pools {
    if pool < 0 || pool >= pooled.pool_count {
      raise IndexOutOfBounds
    }
    if seen.contains(pool) {
      raise DuplicateIndex
    }
    seen[pool] = ()
    for component in 0..<pooled.head_dimension {
      candidate_keys.push(
        Float::from_double(
          pooled.keys[pool * pooled.head_dimension + component],
        ),
      )
    }
  }
  let query : Array[Float] = [
    for value in query_heads => Float::from_double(value)
  ]
  let weights : Array[Float] = [
    for value in head_weights => Float::from_double(value)
  ]
  let selected = @advanced_decoder_reference.learned_index_top_k(
    query,
    candidate_keys,
    weights,
    head_count~,
    head_dimension=pooled.head_dimension,
    top_k=top_k_pools,
    limits=advanced_limits,
  ) catch {
    _ => raise SharedPrimitiveFailure
  }
  let raw_indices : Array[Int] = []
  for local_pool in selected.indices() {
    let pool = candidate_pools[local_pool]
    for offset in 0..<pooled.pool_size {
      raw_indices.push(pooled.raw_indices[pool * pooled.pool_size + offset])
    }
  }
  record_dsa_topk(source_layer~, key_count~, indices=raw_indices, limits~)
}

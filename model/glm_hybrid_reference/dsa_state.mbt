///|
/// Records the result returned by the shared stable top-k primitive at a DSA
/// compute layer. `-1` is the sole allowed padding sentinel.
pub fn record_dsa_topk(
  source_layer~ : Int,
  key_count~ : Int,
  indices~ : ArrayView[Int],
  limits~ : GlmHybridReferenceLimits,
) -> GlmDsaTopKState raise GlmHybridReferenceError {
  if source_layer < 0 {
    raise InvalidLayerOrder
  }
  limits.check_sequence(key_count)
  if indices.is_empty() || indices.length() > key_count + 3 {
    raise BoundsExceeded
  }
  let seen : Map[Int, Unit] = Map([])
  for index in indices {
    if index < -1 || index >= key_count {
      raise IndexOutOfBounds
    }
    if index >= 0 {
      if seen.contains(index) {
        raise DuplicateIndex
      }
      seen[index] = ()
    }
  }
  {
    source_layer,
    last_consumer_layer: source_layer,
    key_count,
    indices: ReadOnlyArray::from_array(indices),
  }
}

///|
/// Advances cross-layer IndexShare state exactly one decoder layer. A skipped,
/// repeated, stale, or changed-key-count consumer fails closed.
pub fn reuse_dsa_topk(
  state : GlmDsaTopKState,
  consumer_layer~ : Int,
  key_count~ : Int,
) -> GlmDsaTopKState raise GlmHybridReferenceError {
  if consumer_layer != state.last_consumer_layer + 1 {
    raise InvalidLayerOrder
  }
  if key_count != state.key_count {
    raise StaleDsaState
  }
  {
    source_layer: state.source_layer,
    last_consumer_layer: consumer_layer,
    key_count,
    indices: state.indices,
  }
}

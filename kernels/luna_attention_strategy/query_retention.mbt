///|
/// Enumerate invariant operand lifetimes independently of KV transfer width.
/// AOT measurement decides whether the saved shared loads justify live state.
/// Existing IDs retain their per-fold lifetime and numerical identity.
fn retained_query_attention_candidates(
  candidates : Array[AttentionTileCandidate],
) -> Array[AttentionTileCandidate] {
  let variants : Array[AttentionTileCandidate] = []
  for candidate in candidates {
    let (queries, keys, dimensions) = candidate.tile.dimensions()
    if attention_fragment_ownership(
        candidate.compute,
        query_rows=queries,
        key_rows=keys,
        head_columns=dimensions,
        subgroups=candidate.subgroups_per_block,
      ) !=
      QueryOwnedFragments {
      continue
    }
    // Prefix and complete retention are separate live ranges. These ranges
    // are derived from the existing reduction extent, never target names.
    for index, retained in [dimensions / 2, dimensions] {
      variants.push({
        ..candidate,
        stable_id: 10000 + candidate.stable_id * 2 + index,
        retained_query_dimensions: retained,
      })
      variants.push({
        ..candidate,
        stable_id: 20000 + candidate.stable_id * 2 + index,
        retained_query_dimensions: retained,
        joint_query_consumers: true,
      })
    }
  }
  variants
}

///|
pub fn AttentionTileCandidate::joint_query_consumers(
  self : AttentionTileCandidate,
) -> Bool {
  self.joint_query_consumers
}

///|
pub fn AttentionTileCandidate::retained_query_dimensions(
  self : AttentionTileCandidate,
) -> Int {
  self.retained_query_dimensions
}

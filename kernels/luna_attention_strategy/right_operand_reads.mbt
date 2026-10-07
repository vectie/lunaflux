///|
/// Physical read ownership and read-ahead are explicit alternatives. No
/// unmeasured speedup is assigned; static ties keep the historical candidate.
fn contiguous_right_read_candidates(
  candidates : Array[AttentionTileCandidate],
) -> Array[AttentionTileCandidate] {
  let variants : Array[AttentionTileCandidate] = []
  for candidate in candidates {
    let (q, k, d) = candidate.tile.dimensions()
    if attention_fragment_ownership(
        candidate.compute,
        query_rows=q,
        key_rows=k,
        head_columns=d,
        subgroups=candidate.subgroups_per_block,
      ) !=
      QueryOwnedFragments {
      continue
    }
    for slots in [1, 2] {
      variants.push({
        ..candidate,
        stable_id: (slots + 1) * 100000 + candidate.stable_id,
        right_operand_read_slots: slots,
      })
    }
  }
  variants
}

///|
pub fn AttentionTileCandidate::right_operand_read_slots(
  self : AttentionTileCandidate,
) -> Int {
  self.right_operand_read_slots
}

///|
/// Cross-operation ownership, not a device instruction choice. A query owner
/// retains QK scores, the softmax fold and PV output across the entire key
/// traversal. No score/probability exchange is required between owners.
pub using @luna_attention_strategy {
  type AttentionFragmentOwnership,
  attention_fragment_ownership,
}

///|
pub fn AttentionTileSchedule::fragment_ownership(
  self : AttentionTileSchedule,
) -> AttentionFragmentOwnership {
  let key_rows = for combinator in self.loops {
    if combinator.axis == AttentionKeyValueAxis {
      break combinator.tile
    }
  } nobreak {
    0
  }
  attention_fragment_ownership(
    self.compute,
    query_rows=self.loops[0].tile,
    key_rows~,
    head_columns=self.loops[self.loops.length() - 1].extent,
    subgroups=self.subgroups_per_block,
  )
}

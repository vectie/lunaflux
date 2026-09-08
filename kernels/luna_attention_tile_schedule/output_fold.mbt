///|
/// Storage realization of the output carried by the ordered attention fold.
/// Subgroup ownership preserves the arithmetic order; it changes no semantics.
pub(all) enum AttentionOutputFoldStorage {
  SharedOutputFold
  SubgroupResidentOutputFold
} derive(Debug, Eq)

///|
fn output_fold_storage(
  compute : @luna_attention_strategy.AttentionTileCompute,
  query_tile : Int,
  head_dimension : Int,
  subgroups : Int,
) -> AttentionOutputFoldStorage {
  // Bound the live accumulator to two matrix-row fragments per subgroup.
  // Wider query tiles retain the shared realization until a measured resource
  // model supports their longer register lifetimes.
  if compute == AttentionMatrixQkPv &&
    query_tile > 0 &&
    head_dimension > 0 &&
    query_tile <= 32 &&
    query_tile % 16 == 0 &&
    head_dimension % 16 == 0 &&
    head_dimension / 16 <= subgroups {
    SubgroupResidentOutputFold
  } else {
    SharedOutputFold
  }
}

///|
pub fn AttentionTileSchedule::output_fold_storage(
  self : AttentionTileSchedule,
) -> AttentionOutputFoldStorage {
  output_fold_storage(
    self.compute,
    self.loops[0].tile,
    self.loops[self.loops.length() - 1].extent,
    self.subgroups_per_block,
  )
}

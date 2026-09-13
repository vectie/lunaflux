///|
pub(all) enum AttentionFragmentOwnership {
  ExchangedScoreTiles
  QueryOwnedFragments
} derive(Debug, Eq)

///|
/// A complete query microtile stays with one subgroup across the ordered
/// QK/softmax/PV fold. This is a portable ownership choice, not a CUDA opcode.
pub fn attention_fragment_ownership(
  compute : AttentionTileCompute,
  query_rows~ : Int,
  key_rows~ : Int,
  head_columns~ : Int,
  subgroups~ : Int,
) -> AttentionFragmentOwnership {
  if compute == AttentionMatrixQkPv &&
    (query_rows == 32 || query_rows == 64) &&
    (key_rows == 32 || key_rows == 64) &&
    (head_columns == 64 || head_columns == 128) &&
    subgroups == query_rows / 16 {
    QueryOwnedFragments
  } else {
    ExchangedScoreTiles
  }
}

///|
pub(all) enum AttentionFragmentOwnership {
  ExchangedScoreTiles
  QueryOwnedFragments
} derive(Debug, Eq)

///|
pub extend AttentionFragmentOwnership with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend AttentionFragmentOwnership with Eq::{not_equal, equal}

///|
/// A complete query microtile stays with one subgroup across the ordered
/// QK/softmax/PV fold. This is a portable ownership choice, not a CUDA opcode.
pub fn attention_fragment_ownership(
  compute : AttentionTileCompute,
  query_rows~ : Int32,
  key_rows~ : Int32,
  head_columns~ : Int32,
  subgroups~ : Int32,
) -> AttentionFragmentOwnership {
  if (compute == AttentionMatrixQkPv || compute == AttentionMatrixQkPvApproxExp) &&
    (query_rows == 32 || query_rows == 64) &&
    (key_rows == 32 || key_rows == 64 || key_rows == 128) &&
    (head_columns == 64 || head_columns == 128) &&
    subgroups == query_rows / 16 {
    QueryOwnedFragments
  } else {
    ExchangedScoreTiles
  }
}

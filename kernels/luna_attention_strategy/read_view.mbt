///|
/// Realizations of the same immutable logical KV stream. Dense-current views
/// require current rows to contain exactly the K/V values committed to the
/// cache (including normalization/position transforms). It does not authorize
/// skipping the cache write needed by future requests.
pub(all) enum AttentionKeyValueReadView {
  PagedCacheOnly
  DenseCurrentAndPagedHistory
  /// The producer additionally guarantees position(row, i) = start(row) + i.
  /// This is a semantic precondition, not a shape heuristic. Frontends with
  /// arbitrary positions must retain DenseCurrentAndPagedHistory.
  AffineDenseCurrentAndPagedHistory
} derive(Debug, Eq)

///|
pub extend AttentionKeyValueReadView with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend AttentionKeyValueReadView with Eq::{not_equal, equal}

///|
pub(all) enum AttentionKeyValueReadAddress {
  CurrentRow(Int32)
  HistoricalPosition(Int32)
} derive(Debug, Eq)

///|
pub extend AttentionKeyValueReadAddress with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend AttentionKeyValueReadAddress with Eq::{not_equal, equal}

///|
/// Pure interval substitution. A backend must additionally check the actual
/// current-row position when a producer permits non-contiguous positions.
pub fn attention_key_value_read_address(
  view : AttentionKeyValueReadView,
  position~ : Int32,
  current_begin~ : Int32,
  current_count~ : Int32,
) -> AttentionKeyValueReadAddress? {
  if position < 0 ||
    current_begin < 0 ||
    current_count < 0 ||
    current_begin > 2147483647 - current_count {
    return None
  }
  if view != PagedCacheOnly &&
    position >= current_begin &&
    position - current_begin < current_count {
    Some(CurrentRow(position - current_begin))
  } else {
    Some(HistoricalPosition(position))
  }
}

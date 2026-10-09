///|
/// Immutable storage/effect plan for one request's projected sparse history.
/// Frame rows are appended contiguously; request reset replaces the history.
/// These BF16 bit-copy effects are independent of attention arithmetic and GPU.
pub struct IndexedSparseHistoryPrecision {
  priv attention : IndexedSparseAttentionPrecision
  priv sizes : ReadOnlyArray[Int64]
  priv bytes : Int64
}

///|
pub fn IndexedSparseHistoryPrecision::new(
  attention : IndexedSparseAttentionPrecision,
  maximum_state_bytes~ : Int64,
) -> Self raise PrecisionPlanError {
  let (rows, history) = attention.capacity()
  let (_, dimension, _, _, _) = attention.index_geometry()
  let (_, heads, keys, values) = attention.attention_geometry()
  let h = history.to_int64()
  let sizes = [
    expert_checked_mul(expert_checked_mul(h, dimension.to_int64()), 2L),
    expert_checked_mul(expert_checked_mul(h, dimension.to_int64()), 2L),
    expert_checked_mul(
      expert_checked_mul(
        expert_checked_mul(h, heads.to_int64()),
        keys.to_int64(),
      ),
      2L,
    ),
    expert_checked_mul(
      expert_checked_mul(
        expert_checked_mul(h, heads.to_int64()),
        values.to_int64(),
      ),
      2L,
    ),
    h * 4L,
    20L,
    12L,
    rows.to_int64() * 4L,
  ]
  // Attention geometry bounds ensure this eight-buffer sum fits Int64.
  let bytes = sizes.fold(init=0L, (sum, size) => sum + size)
  guard bytes <= maximum_state_bytes else { raise MemoryBudgetExceeded }
  { attention, sizes: ReadOnlyArray::from_array(sizes), bytes, }
}

///|
pub fn IndexedSparseHistoryPrecision::attention(
  self : Self,
) -> IndexedSparseAttentionPrecision {
  self.attention
}

///|
/// Index keys, pool gates, attention keys/values, valid mask, counts, append
/// descriptor, and absolute query positions, in that order.
pub fn IndexedSparseHistoryPrecision::buffer_bytes(
  self : Self,
) -> ReadOnlyArray[Int64] {
  self.sizes
}

///|
pub fn IndexedSparseHistoryPrecision::state_bytes(self : Self) -> Int64 {
  self.bytes
}

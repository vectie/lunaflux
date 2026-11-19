///|
pub(all) enum KvWriteEffect {
  StoreKey
  StoreValue
} derive(Eq, Debug)

///|
pub extend KvWriteEffect with Eq::{equal, not_equal}

///|
pub extend KvWriteEffect with @debug.Debug::{to_repr}

///|
/// Ordered writes into disjoint split-K/V arenas from packed Q/K/V rows.
/// Page lookup remains an explicit runtime input, never a compiler-owned cache.
pub struct KvWriteProgram {
  priv query_width : Int32
  priv kv_width : Int32
  priv page_stride_elements : Int64
  priv ownership : @physical_tile_ir.VectorOwnership
} derive(Eq, Debug)

///|
pub extend KvWriteProgram with Eq::{equal, not_equal}

///|
pub extend KvWriteProgram with @debug.Debug::{to_repr}

///|
pub fn KvWriteProgram::new(
  query_heads~ : Int32,
  kv_heads~ : Int32,
  head_dimension~ : Int32,
  page_stride_bytes~ : Int64,
  element_bytes~ : Int32,
  owners~ : Int32,
) -> KvWriteProgram? {
  guard query_heads > 0 &&
    kv_heads > 0 &&
    head_dimension > 0 &&
    element_bytes > 0 &&
    page_stride_bytes > 0L &&
    page_stride_bytes % element_bytes.to_int64() == 0L else {
    return None
  }
  let q = query_heads.to_int64() * head_dimension.to_int64()
  let kv = kv_heads.to_int64() * head_dimension.to_int64()
  guard q <= 2147483647L && kv <= (2147483647L - q) / 2L else { return None }
  guard @physical_tile_ir.VectorOwnership::new(vectors=kv.to_int(), owners~)
    is Some(ownership) else {
    return None
  }
  Some({
    query_width: q.to_int(),
    kv_width: kv.to_int(),
    page_stride_elements: page_stride_bytes / element_bytes.to_int64(),
    ownership,
  })
}

///|
pub fn KvWriteProgram::geometry(self : KvWriteProgram) -> (Int32, Int32, Int64) {
  (self.query_width, self.kv_width, self.page_stride_elements)
}

///|
pub fn KvWriteProgram::ownership(
  self : KvWriteProgram,
) -> @physical_tile_ir.VectorOwnership {
  self.ownership
}

///|
pub fn KvWriteProgram::effects(
  _self : KvWriteProgram,
) -> ReadOnlyArray[KvWriteEffect] {
  [StoreKey, StoreValue]
}

///|
pub(all) enum IngressEffect {
  LoadHead
  PublishHead
  NormalizeRotateQueryKey
  StoreActivationAndCache
  RetireHead
} derive(Eq, Debug)

///|
pub extend IngressEffect with Eq::{equal, not_equal}

///|
pub extend IngressEffect with @debug.Debug::{to_repr}

///|
/// One cooperative owner group per head. Values bypass normalization and
/// rotation. Publication precedes cross-owner pair reads, and retirement keeps
/// the next token from overwriting scratch before stores have consumed it.
pub struct IngressProgram {
  priv query_heads : Int
  priv kv_heads : Int
  priv head_dimension : Int
  priv ownership : @physical_tile_ir.VectorOwnership
} derive(Eq, Debug)

///|
pub extend IngressProgram with Eq::{equal, not_equal}

///|
pub extend IngressProgram with @debug.Debug::{to_repr}

///|
pub fn IngressProgram::new(
  query_heads~ : Int,
  kv_heads~ : Int,
  head_dimension~ : Int,
  owners~ : Int,
) -> IngressProgram? {
  guard query_heads > 0 &&
    kv_heads > 0 &&
    head_dimension > 0 &&
    head_dimension % 2 == 0 else {
    return None
  }
  let packed = query_heads.to_int64() + 2L * kv_heads.to_int64()
  guard packed <= 2147483647L / head_dimension.to_int64() else { return None }
  guard @physical_tile_ir.VectorOwnership::new(vectors=head_dimension, owners~)
    is Some(ownership) else {
    return None
  }
  Some({ query_heads, kv_heads, head_dimension, ownership, })
}

///|
pub fn IngressProgram::geometry(self : IngressProgram) -> (Int, Int, Int) {
  (
    self.query_heads + 2 * self.kv_heads,
    self.query_heads + self.kv_heads,
    self.kv_heads * self.head_dimension,
  )
}

///|
pub fn IngressProgram::ownership(
  self : IngressProgram,
) -> @physical_tile_ir.VectorOwnership {
  self.ownership
}

///|
pub fn IngressProgram::effects(
  _self : IngressProgram,
) -> ReadOnlyArray[IngressEffect] {
  [
    LoadHead,
    PublishHead,
    NormalizeRotateQueryKey,
    StoreActivationAndCache,
    RetireHead,
  ]
}

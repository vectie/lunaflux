///|
/// A complete query's head-by-candidate product. Each tile owns all heads
/// for its candidate interval, so no atomics or inter-tile partial sum exists.
/// Matrix fragment geometry belongs to device lowering, not this ownership IR.
pub struct MatrixScorePlan {
  priv tiles : CandidateScorePlan
  priv heads : Int
  priv dimension : Int
}

///|
pub fn MatrixScorePlan::new(
  semantic : @precision.IndexedSparseAttentionPrecision,
  candidate_width~ : Int,
) -> Self? {
  guard semantic.index_scoring() is RowScaledE4M3Matrix && candidate_width > 0 else {
    return None
  }
  let (heads, dimension, _, _, _) = semantic.index_geometry()
  let retained = RetainedIndexPlan::new(semantic)
  // Unlike the scalar splitter, a matrix tile remains required for a short
  // static domain. It owns masked padding, never synthetic valid candidates.
  let (queries, candidates) = retained.geometry()
  let tiles = (candidates.to_int64() + candidate_width.to_int64() - 1L) /
    candidate_width.to_int64()
  guard tiles * queries.to_int64() <= 2147483647L else { return None }
  Some({
    tiles: { queries, candidates, width: candidate_width, },
    heads,
    dimension,
  })
}

///|
pub fn MatrixScorePlan::tiles(self : Self) -> CandidateScorePlan {
  self.tiles
}

///|
pub fn MatrixScorePlan::operand_bytes(self : Self) -> (Int64, Int64) {
  (
    self.heads.to_int64() * self.dimension.to_int64(),
    self.tiles.width.to_int64() * (self.dimension.to_int64() + 4L),
  )
}

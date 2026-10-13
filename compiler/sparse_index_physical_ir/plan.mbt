///|
/// Independent query/candidate ownership with one retained ordered score and
/// one availability flag per candidate. Selection consumes scores, never
/// recomputes their dot products. No device names or launch dimensions here.
pub struct RetainedIndexPlan {
  priv queries : Int
  priv candidates : Int
}

///|
pub fn RetainedIndexPlan::new(
  semantic : @precision.IndexedSparseAttentionPrecision,
) -> Self {
  let (queries, history) = semantic.capacity()
  let (_, _, pool, _, _) = semantic.index_geometry()
  { queries, candidates: history / pool, }
}

///|
pub fn RetainedIndexPlan::geometry(self : Self) -> (Int, Int) {
  (self.queries, self.candidates)
}

///|
pub fn RetainedIndexPlan::scratch_bytes(self : Self) -> Int64 {
  self.queries.to_int64() * self.candidates.to_int64() * 8L
}

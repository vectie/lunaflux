///|
/// Independent query/candidate ownership with one retained ordered score and
/// one availability flag per candidate. Selection consumes scores, never
/// recomputes their dot products. No device names or launch dimensions here.
pub struct RetainedIndexPlan {
  priv queries : Int32
  priv candidates : Int32
  priv published_slots : Int32
  priv selected_pools : Int32
}

///|
pub fn RetainedIndexPlan::new(
  semantic : @precision.IndexedSparseAttentionPrecision,
) -> Self {
  let (queries, history) = semantic.capacity()
  let (_, _, pool, top_k, tail) = semantic.index_geometry()
  {
    queries,
    candidates: history / pool,
    selected_pools: top_k / pool,
    published_slots: Int32::min(history / pool, top_k / pool) * pool +
    (if tail { pool - 1 } else { 0 }),
  }
}

///|
/// The producer writes selected pools followed by the incomplete tail. All
/// slots beyond this prefix are padding for every admitted history. Consumers
/// may restrict traversal only when bound to this producer, not arbitrary lists.
pub fn RetainedIndexPlan::published_slots(self : Self) -> Int32 {
  self.published_slots
}

///|
pub fn RetainedIndexPlan::geometry(self : Self) -> (Int32, Int32) {
  (self.queries, self.candidates)
}

///|
pub fn RetainedIndexPlan::scratch_bytes(self : Self) -> Int64 {
  self.queries.to_int64() * self.candidates.to_int64() * 8L
}

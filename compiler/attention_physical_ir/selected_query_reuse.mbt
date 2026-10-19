///|
/// A query cohort shares history reads, not sparse membership. Each row keeps
/// its own mask and causal boundary; cohorts containing multiple sequences are
/// partitioned by sequence at execution. Scores have a separate publication
/// epoch before normalized probabilities reuse their storage.
pub struct SelectedQueryReuse {
  priv queries : Int
  priv heads : Int
  priv history : Int
  priv rows : Int
}

///|
pub fn SelectedQueryReuse::new(
  queries~ : Int,
  heads~ : Int,
  history~ : Int,
  query_dimension~ : Int,
  value_dimension~ : Int,
  rows~ : Int,
  shared_budget~ : Int64,
) -> Self? {
  guard queries >= rows &&
    heads > 0 &&
    history > 0 &&
    rows == 16 &&
    query_dimension > 0 &&
    query_dimension % 16 == 0 &&
    value_dimension > 0 &&
    value_dimension % 8 == 0 else {
    return None
  }
  let words = (history.to_int64() + 31L) / 32L
  let shared = rows.to_int64() * words * 4L +
    rows.to_int64() *
    Int::max(query_dimension, value_dimension).to_int64() *
    2L +
    4096L
  guard shared <= shared_budget &&
    queries.to_int64() * heads.to_int64() <= 2147483647L &&
    queries.to_int64() * heads.to_int64() <=
    9223372036854775807L / history.to_int64() / 4L else {
    return None
  }
  Some({ queries, heads, history, rows, })
}

///|
pub fn SelectedQueryReuse::workgroups(self : Self) -> Int {
  ((self.queries - 1) / self.rows + 1) * self.heads
}

///|
pub fn SelectedQueryReuse::cohort_rows(self : Self) -> Int {
  self.rows
}

///|
/// The dispatch covers both row-wise decode and cohort prefill ownership.
pub fn SelectedQueryReuse::dispatch_workgroups(self : Self) -> Int {
  self.queries * self.heads
}

///|
pub fn SelectedQueryReuse::scratch_bytes(self : Self) -> Int64 {
  self.queries.to_int64() * self.heads.to_int64() * self.history.to_int64() * 4L
}

///|
pub fn SelectedQueryReuse::owns_sequence(
  self : Self,
  sequences : ArrayView[Int],
  row : Int,
) -> Bool {
  guard row >= 0 &&
    row < sequences.length() &&
    row < self.rows &&
    sequences[row] >= 0 else {
    return false
  }
  for earlier in 0..<row {
    if sequences[earlier] == sequences[row] {
      return false
    }
  }
  true
}

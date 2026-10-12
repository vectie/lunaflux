///|
/// Stable materialization of selected keys and scores. Each query/head owns a
/// disjoint region; workers map independent scores or value columns, never
/// split an ordered numeric reduction. Units are portable logical workers,
/// not CUDA warps. Sentinel filtering preserves selected-rank order.
pub struct SelectedSparseSchedule {
  priv groups : Int
  priv slots : Int
  priv workers : Int
  priv scratch : Int64
} derive(Debug, Eq)

///|
pub extend SelectedSparseSchedule with Eq::{equal, not_equal}

///|
pub extend SelectedSparseSchedule with @debug.Debug::{to_repr}

///|
pub fn SelectedSparseSchedule::new(
  queries~ : Int,
  heads~ : Int,
  selected_width~ : Int,
  history~ : Int,
  workers~ : Int,
) -> Self? {
  guard queries > 0 &&
    heads > 0 &&
    selected_width > 0 &&
    history > 0 &&
    workers > 0 &&
    queries.to_int64() * heads.to_int64() <= 2147483647L else {
    return None
  }
  let groups = queries * heads
  let slots = Int::min(selected_width, history)
  guard groups.to_int64() <= 9223372036854775807L / 8L / slots.to_int64() else {
    return None
  }
  Some({
    groups,
    slots,
    workers,
    scratch: groups.to_int64() * slots.to_int64() * 8L,
  })
}

///|
pub fn SelectedSparseSchedule::workgroups(self : Self) -> Int {
  self.groups
}

///|
pub fn SelectedSparseSchedule::workers(self : Self) -> Int {
  self.workers
}

///|
pub fn SelectedSparseSchedule::retained_slots(self : Self) -> Int {
  self.slots
}

///|
pub fn SelectedSparseSchedule::scratch_bytes(self : Self) -> Int64 {
  self.scratch
}

///|
/// Keys and scores are separate regions; a score becomes a probability only
/// after maximum/denominator consumers finish. Offsets are in four-byte cells.
pub fn SelectedSparseSchedule::region_cells(
  self : Self,
  group : Int,
) -> (Int64, Int64)? {
  guard group >= 0 && group < self.groups else { return None }
  let base = group.to_int64() * self.slots.to_int64() * 2L
  Some((base, base + self.slots.to_int64()))
}

///|
pub fn SelectedSparseSchedule::independent_owner(
  self : Self,
  item : Int,
) -> Int? {
  if item < 0 {
    None
  } else {
    Some(item % self.workers)
  }
}

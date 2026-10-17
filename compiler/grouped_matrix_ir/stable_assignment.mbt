///|
/// First-occurrence grouping is factored into independent ranks and one
/// ordered integer prefix. This changes work distribution, never job order.
pub struct StableAssignmentPlan {
  priv capacity : Int
  priv rows_per_group : Int
}

///|
pub fn ExpertGroupingPlan::stable_assignment(
  self : Self,
) -> StableAssignmentPlan {
  { capacity: self.maximum_groups(), rows_per_group: self.rows_per_group(), }
}

///|
pub fn StableAssignmentPlan::capacity(self : Self) -> Int {
  self.capacity
}

///|
pub fn StableAssignmentPlan::rows_per_group(self : Self) -> Int {
  self.rows_per_group
}

///|
/// Three integer arrays: first occurrence, within-expert rank, prefix offsets.
pub fn StableAssignmentPlan::scratch_elements(self : Self) -> Int {
  self.capacity * 3
}

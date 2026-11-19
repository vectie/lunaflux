///|
/// Prefix validity of instruction row groups. Padded lanes of the final live
/// group remain part of the instruction; wholly inactive groups have no reads,
/// products or publications. This is independent of lane and device geometry.
pub struct FragmentRowDomain {
  priv groups : Int32
  priv quantum : Int32
} derive(Eq, Debug)

///|
pub extend FragmentRowDomain with Eq::{equal, not_equal}

///|
pub extend FragmentRowDomain with @debug.Debug::{to_repr}

///|
pub fn FragmentProgram::row_domain(
  self : FragmentProgram,
  quantum~ : Int32,
) -> FragmentRowDomain? {
  guard quantum > 0 && self.rows <= 2147483647 / quantum else { return None }
  Some({ groups: self.rows, quantum, })
}

///|
pub fn FragmentRowDomain::geometry(self : FragmentRowDomain) -> (Int32, Int32) {
  (self.groups, self.quantum)
}

///|
/// The caller supplies the remaining logical prefix, not a rounded tile size.
/// Subtraction before ceiling division avoids overflow at the largest extent.
pub fn FragmentRowDomain::active_groups(
  self : FragmentRowDomain,
  remaining : Int32,
) -> Int32 {
  if remaining <= 0 {
    0
  } else {
    let live = 1 + (remaining - 1) / self.quantum
    if live < self.groups {
      live
    } else {
      self.groups
    }
  }
}

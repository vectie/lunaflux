///|
/// Placement of a descending reduction tree, not a new arithmetic order.
/// Workgroup levels precede a single subgroup-owned suffix. The device
/// supplies its subgroup width; numerical IR continues to own the leaves.
pub struct ReductionTreePartition {
  priv shared_levels : ReadOnlyArray[Int32]
  priv local_levels : ReadOnlyArray[Int32]
}

///|
pub fn ReductionTreePartition::new(
  workers : Int32,
  subgroup_width : Int32,
) -> Self raise @precision.PrecisionPlanError {
  guard workers > 0 &&
    workers <= 1024 &&
    (workers & (workers - 1)) == 0 &&
    subgroup_width > 0 &&
    subgroup_width <= workers &&
    (subgroup_width & (subgroup_width - 1)) == 0 else {
    raise NoImplementation
  }
  let shared = []
  let local_levels = []
  let mut stride = workers / 2
  while stride > 0 {
    if stride >= subgroup_width {
      shared.push(stride)
    } else {
      local_levels.push(stride)
    }
    stride = stride / 2
  }
  {
    shared_levels: ReadOnlyArray::from_array(shared),
    local_levels: ReadOnlyArray::from_array(local_levels),
  }
}

///|
pub fn ReductionTreePartition::levels(
  self : Self,
) -> (ReadOnlyArray[Int32], ReadOnlyArray[Int32]) {
  (self.shared_levels, self.local_levels)
}

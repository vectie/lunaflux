///|
/// Redistribute an ordered logical-owner tree onto a consumer subgroup without
/// changing its leaves or addition edges. Original producers retain their
/// strided inputs and publish partials once; each consumer subgroup replicates
/// this pure tree suffix, with upper levels as lane-local register additions.
pub struct RetainedNormalizationTree {
  priv logical_owners : Int
  priv lanes : Int
  priv values : Int
} derive(Eq, Debug)

///|
pub extend RetainedNormalizationTree with Eq::{equal, not_equal}

///|
pub extend RetainedNormalizationTree with @debug.Debug::{to_repr}

///|
pub fn NormalizationProgram::retained_tree(
  self : NormalizationProgram,
  maximum_values_per_lane~ : Int,
) -> RetainedNormalizationTree? {
  guard self.owner_tree && self.residual && maximum_values_per_lane > 0 else {
    return None
  }
  let values = (self.width - 1) / self.subgroup_lanes + 1
  guard values <= maximum_values_per_lane else { return None }
  Some({ logical_owners: self.owners, lanes: self.subgroup_lanes, values, })
}

///|
pub fn RetainedNormalizationTree::geometry(
  self : RetainedNormalizationTree,
) -> (Int, Int, Int) {
  (self.logical_owners, self.lanes, self.values)
}

///|
/// Each pair is an original logical-tree edge within one physical lane.
pub fn RetainedNormalizationTree::local_edges(
  self : RetainedNormalizationTree,
) -> ReadOnlyArray[(Int, Int)] {
  let edges = []
  for stride = self.logical_owners / self.lanes / 2
      stride > 0
      stride = stride / 2 {
    for destination in 0..<stride {
      edges.push((destination, destination + stride))
    }
  }
  ReadOnlyArray::from_array(edges)
}

///|
pub fn RetainedNormalizationTree::exchange_strides(
  self : RetainedNormalizationTree,
) -> ReadOnlyArray[Int] {
  let strides = []
  for stride = self.lanes / 2; stride > 0; stride = stride / 2 {
    strides.push(stride)
  }
  ReadOnlyArray::from_array(strides)
}

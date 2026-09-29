///|
pub(all) enum SelectionReduction {
  OwnerTree
  SubgroupTree
} derive(Eq, Debug)

///|
pub extend SelectionReduction with Eq::{equal, not_equal}

///|
pub extend SelectionReduction with @debug.Debug::{to_repr}

///|
pub(all) enum SelectionEffect {
  ScanOwnedValues
  PublishOwnerSummaries
  SynchronizeSelection
  ReduceOwnerSummaries(Int)
  ReduceSubgroupSummaries(Int)
  PublishSubgroupSummaries
  ReducePublishedSubgroups(Int)
  StoreSelection
} derive(Eq, Debug)

///|
pub extend SelectionEffect with Eq::{equal, not_equal}

///|
pub extend SelectionEffect with @debug.Debug::{to_repr}

///|
/// Greatest finite value, lowest token on ties, and lowest invalid token form
/// one associative summary. Unlike softmax, this fold permits partitioning
/// without reassociating a floating-point sum.
pub struct SelectionProgram {
  priv ownership : @physical_tile_ir.VectorOwnership
  priv subgroup_lanes : Int
  priv reduction : SelectionReduction
} derive(Eq, Debug)

///|
pub extend SelectionProgram with Eq::{equal, not_equal}

///|
pub extend SelectionProgram with @debug.Debug::{to_repr}

///|
pub fn SelectionProgram::new(
  values~ : Int,
  owners~ : Int,
  subgroup_lanes~ : Int,
  reduction~ : SelectionReduction,
) -> SelectionProgram? {
  guard owners > 0 &&
    (owners & (owners - 1)) == 0 &&
    subgroup_lanes > 0 &&
    (subgroup_lanes & (subgroup_lanes - 1)) == 0 &&
    owners % subgroup_lanes == 0 &&
    owners / subgroup_lanes <= subgroup_lanes else {
    return None
  }
  guard @physical_tile_ir.VectorOwnership::new(vectors=values, owners~)
    is Some(ownership) else {
    return None
  }
  Some({ ownership, subgroup_lanes, reduction, })
}

///|
pub fn SelectionProgram::geometry(self : SelectionProgram) -> (Int, Int, Int) {
  let (values, owners, _) = self.ownership.geometry()
  (values, owners, self.subgroup_lanes)
}

///|
pub fn SelectionProgram::effects(
  self : SelectionProgram,
) -> ReadOnlyArray[SelectionEffect] {
  let (_, owners, lanes) = self.geometry()
  match self.reduction {
    OwnerTree =>
      [
        ScanOwnedValues,
        PublishOwnerSummaries,
        SynchronizeSelection,
        ReduceOwnerSummaries(owners / 2),
        StoreSelection,
      ]
    SubgroupTree =>
      [
        ScanOwnedValues,
        ReduceSubgroupSummaries(lanes / 2),
        PublishSubgroupSummaries,
        SynchronizeSelection,
        ReducePublishedSubgroups(lanes / 2),
        StoreSelection,
      ]
  }
}

///|
pub fn SelectionProgram::prefers(
  _self : SelectionProgram,
  value : Double,
  token : Int,
  best : Double,
  best_token : Int,
) -> Bool {
  value > best || (value == best && token < best_token)
}

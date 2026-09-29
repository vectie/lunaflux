///|
pub(all) enum NormalizationPrepare {
  LoadInput
  AddResidual
  RoundBf16
  AccumulateSquare
} derive(Eq, Debug)

///|
pub extend NormalizationPrepare with Eq::{equal, not_equal}

///|
pub extend NormalizationPrepare with @debug.Debug::{to_repr}

///|
pub(all) enum ReductionEffect {
  ReduceSubgroup
  PublishPartials
  SynchronizeOwners
  ReducePartials
  PublishTotal
  ReduceOwnerTree(Int)
} derive(Eq, Debug)

///|
pub extend ReductionEffect with Eq::{equal, not_equal}

///|
pub extend ReductionEffect with @debug.Debug::{to_repr}

///|
/// A row's ordered sum-of-squares reduction. Residual addition is rounded to
/// BF16 before reduction, matching the unfused semantic boundary. The target
/// supplies subgroup width; no device instruction or fixed lane count lives here.
pub struct NormalizationProgram {
  priv width : Int
  priv owners : Int
  priv subgroup_lanes : Int
  priv residual : Bool
  priv owner_tree : Bool
} derive(Eq, Debug)

///|
pub extend NormalizationProgram with Eq::{equal, not_equal}

///|
pub extend NormalizationProgram with @debug.Debug::{to_repr}

///|
pub fn NormalizationProgram::new(
  width~ : Int,
  owners~ : Int,
  subgroup_lanes~ : Int,
  residual~ : Bool,
  owner_tree? : Bool = false,
) -> NormalizationProgram? {
  guard width > 0 &&
    owners > 0 &&
    subgroup_lanes > 0 &&
    (subgroup_lanes & (subgroup_lanes - 1)) == 0 &&
    owners % subgroup_lanes == 0 &&
    (if owner_tree {
      (owners & (owners - 1)) == 0
    } else {
      owners / subgroup_lanes <= subgroup_lanes
    }) else {
    return None
  }
  Some({ width, owners, subgroup_lanes, residual, owner_tree, })
}

///|
pub fn NormalizationProgram::geometry(
  self : NormalizationProgram,
) -> (Int, Int, Int) {
  (self.width, self.owners, self.subgroup_lanes)
}

///|
pub fn NormalizationProgram::partial_count(self : NormalizationProgram) -> Int {
  if self.owner_tree {
    self.owners
  } else {
    self.owners / self.subgroup_lanes
  }
}

///|
pub fn NormalizationProgram::values_per_owner(
  self : NormalizationProgram,
) -> Int {
  (self.width - 1) / self.owners + 1
}

///|
pub fn NormalizationProgram::preparation(
  self : NormalizationProgram,
) -> ReadOnlyArray[NormalizationPrepare] {
  if self.residual {
    [LoadInput, AddResidual, RoundBf16, AccumulateSquare]
  } else {
    [LoadInput, AccumulateSquare]
  }
}

///|
pub fn NormalizationProgram::reduction(
  self : NormalizationProgram,
) -> ReadOnlyArray[ReductionEffect] {
  if self.owner_tree {
    [PublishPartials, SynchronizeOwners, ReduceOwnerTree(self.owners / 2)]
  } else if self.partial_count() == 1 {
    [ReduceSubgroup]
  } else {
    [
      ReduceSubgroup,
      PublishPartials,
      SynchronizeOwners,
      ReducePartials,
      PublishTotal,
      SynchronizeOwners,
    ]
  }
}

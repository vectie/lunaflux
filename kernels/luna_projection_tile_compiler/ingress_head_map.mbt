///|
/// Pack independent head folds into a workgroup without changing any fold's
/// reduction tree. A subgroup owns one complete head, including its terminal
/// write; tail subgroups have no head and therefore no effects.
pub struct AttentionIngressHeadMap {
  priv heads : Int
  priv lanes_per_head : Int
  priv heads_per_workgroup : Int
} derive(Debug, Eq)

///|
pub fn AttentionIngressNumerics::group_heads(
  self : AttentionIngressNumerics,
  heads~ : Int,
  workgroup_lanes~ : Int,
) -> AttentionIngressHeadMap raise ProjectionTileCompilerError {
  guard heads > 0 &&
    workgroup_lanes >= self.reduction_lanes &&
    workgroup_lanes <= 1024 &&
    workgroup_lanes % self.reduction_lanes == 0 else {
    raise InvalidProjectionTileCompilation(InvalidProblem)
  }
  {
    heads,
    lanes_per_head: self.reduction_lanes,
    heads_per_workgroup: workgroup_lanes / self.reduction_lanes,
  }
}

///|
pub fn AttentionIngressHeadMap::geometry(
  self : AttentionIngressHeadMap,
) -> (Int, Int, Int) {
  (
    (self.heads - 1) / self.heads_per_workgroup + 1,
    self.heads_per_workgroup,
    self.lanes_per_head,
  )
}

///|
pub fn AttentionIngressHeadMap::head_at(
  self : AttentionIngressHeadMap,
  workgroup : Int,
  subgroup : Int,
) -> Int? {
  let (groups, _, _) = self.geometry()
  guard workgroup >= 0 &&
    workgroup < groups &&
    subgroup >= 0 &&
    subgroup < self.heads_per_workgroup else {
    return None
  }
  // The final partial workgroup must not overflow even for a maximal domain.
  let base = workgroup * self.heads_per_workgroup
  if subgroup < self.heads - base {
    Some(base + subgroup)
  } else {
    None
  }
}

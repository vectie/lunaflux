///|
/// One executable region and its complete live-boundary contracts. Class
/// equality includes layout, numerics and ownership, not just tensor shape.
/// The catalog owner interns these contracts; this IR cannot authorize a cast.
pub struct RegionImplementation {
  priv id : Int
  priv region : Region
  priv input : ContinuationClass
  priv output : ContinuationClass
  priv workspace_bytes : Int64
  priv local_bytes : Int64?
}

///|
pub fn RegionImplementation::new(
  id~ : Int,
  region : Region,
  input~ : ContinuationClass,
  output~ : ContinuationClass,
  workspace_bytes~ : Int64,
  local_bytes~ : Int64?,
) -> RegionImplementation raise FusionRegionError {
  if id < 0 ||
    !input.valid() ||
    !output.valid() ||
    workspace_bytes < 0L ||
    (local_bytes is Some(bytes) && bytes < 0L) {
    raise InvalidComparison
  }
  { id, region, input, output, workspace_bytes, local_bytes, }
}

///|
/// A connected executable cover, before any local-cost pruning. Persistent
/// live storage is planned separately. Scratch and per-workgroup storage are
/// reusable maxima because these regions execute serially, without overlap.
pub struct ConnectedPlan {
  priv id : Int
  priv partition : Partition
  priv implementations : ReadOnlyArray[Int]
  priv input : ContinuationClass
  priv output : ContinuationClass
  priv workspace_bytes : Int64
  priv local_bytes : Int64?
}

///|
pub fn ConnectedPlan::new(
  id~ : Int,
  graph : Graph,
  regions : ArrayView[RegionImplementation],
) -> ConnectedPlan raise FusionRegionError {
  if id < 0 || regions.is_empty() {
    raise InvalidComparison
  }
  for index in 1..<regions.length() {
    if regions[index - 1].output != regions[index].input {
      raise InvalidComparison
    }
  }
  let partition = Partition::new(graph, regions.map(r => r.region)[:])
  let workspace_bytes = regions.fold(init=0L, (n, r) => n.max(r.workspace_bytes))
  let local_bytes = regions.fold(init=Some(0L), (n, r) => {
    match (n, r.local_bytes) {
      (Some(n), Some(bytes)) => Some(n.max(bytes))
      _ => None
    }
  })
  {
    id,
    partition,
    implementations: ReadOnlyArray::from_array(regions.map(r => r.id)),
    input: regions[0].input,
    output: regions[regions.length() - 1].output,
    workspace_bytes,
    local_bytes,
  }
}

///|
pub fn ConnectedPlan::implementation_ids(
  self : ConnectedPlan,
) -> ReadOnlyArray[Int] {
  self.implementations
}

///|
/// Only complete-chain observations may resolve this catalog. In particular,
/// adding isolated producer and consumer timings is not a chain measurement.
/// Returns the retained executable plan, so binding cannot independently
/// reselect one of its regions after the joint decision.
pub fn select_connected_plan(
  plans : ArrayView[ConnectedPlan],
  costs : ArrayView[Cost],
  maximum_workspace_bytes~ : Int64,
  maximum_local_bytes~ : Int64?,
) -> ConnectedPlan raise FusionRegionError {
  if plans.is_empty() ||
    plans.length() != costs.length() ||
    maximum_workspace_bytes < 0L ||
    (maximum_local_bytes is Some(bytes) && bytes < 0L) {
    raise InvalidComparison
  }
  for index, plan in plans {
    if plan.input != plans[0].input ||
      plan.output != plans[0].output ||
      plan.partition.graph != plans[0].partition.graph ||
      plans[:index].any(prior => prior.id == plan.id) ||
      costs[index].basis() != Measured {
      raise InvalidComparison
    }
  }
  let alternatives = []
  for index, plan in plans {
    if maximum_local_bytes is Some(limit) {
      guard plan.local_bytes is Some(bytes) && bytes <= limit else { continue }
    }
    alternatives.push(
      PartitionAlternative::new(
        id=plan.id,
        plan.partition,
        costs[index],
        workspace_bytes=plan.workspace_bytes,
        // With no local-memory constraint, project this dimension out of the
        // comparison for EVERY plan. Zero here is not a physical resource fact.
        local_bytes=if maximum_local_bytes is Some(_) {
          plan.local_bytes.unwrap()
        } else {
          0L
        },
        continuation_class=plan.output,
      ),
    )
  }
  if alternatives.is_empty() {
    raise NoFeasiblePartition
  }
  let selected = PartitionFrontier::new(alternatives)
    .select(
      continuation_class=plans[0].output,
      maximum_workspace_bytes~,
      maximum_local_bytes=maximum_local_bytes.unwrap_or(0L),
    )
    .implementation_id()
  for plan in plans {
    if plan.id == selected {
      return plan
    }
  }
  raise NoFeasiblePartition
}

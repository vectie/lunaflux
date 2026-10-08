///|
pub(all) enum CostBasis {
  Measured
  Estimated
} derive(Eq, Debug)

///|
pub extend CostBasis with Eq::{equal, not_equal}

///|
pub extend CostBasis with @debug.Debug::{to_repr}

///|
/// Every observation covers the complete region, including launches, boundary
/// materializations and synchronization. Estimates cannot compete with timings.
pub struct Cost {
  priv nanoseconds : Int64
  priv basis : CostBasis
} derive(Eq)

///|
pub extend Cost with Eq::{equal, not_equal}

///|
pub fn Cost::measured(
  nanoseconds~ : Int64,
  samples~ : Int,
) -> Cost raise FusionRegionError {
  if nanoseconds <= 0L || samples < 3 {
    raise InvalidCost
  }
  { nanoseconds, basis: Measured, }
}

///|
pub fn Cost::estimated(nanoseconds~ : Int64) -> Cost raise FusionRegionError {
  if nanoseconds <= 0L {
    raise InvalidCost
  }
  { nanoseconds, basis: Estimated, }
}

///|
pub fn Cost::basis(self : Cost) -> CostBasis {
  self.basis
}

///|
pub fn Cost::nanoseconds(self : Cost) -> Int64 {
  self.nanoseconds
}

///|
pub struct Candidate {
  priv id : Int
  priv region : Region
  priv cost : Cost
  priv workspace_bytes : Int64
  priv local_bytes : Int64
}

///|
pub fn Candidate::new(
  id~ : Int,
  region : Region,
  cost : Cost,
  workspace_bytes~ : Int64,
  local_bytes~ : Int64,
) -> Candidate raise FusionRegionError {
  if id < 0 || workspace_bytes < 0L || local_bytes < 0L {
    raise InvalidCost
  }
  { id, region, cost, workspace_bytes, local_bytes, }
}

///|
pub struct Selection {
  priv partition : Partition
  priv ids : ReadOnlyArray[Int]
  priv cost : Cost
}

///|
pub fn Selection::partition(self : Selection) -> Partition {
  self.partition
}

///|
pub fn Selection::implementation_ids(self : Selection) -> ReadOnlyArray[Int] {
  self.ids
}

///|
pub fn Selection::cost(self : Selection) -> Cost {
  self.cost
}

///|
fn ids_less(left : ArrayView[Int], right : ArrayView[Int]) -> Bool {
  for index in 0..<left.length().min(right.length()) {
    if left[index] != right[index] {
      return left[index] < right[index]
    }
  }
  left.length() < right.length()
}

///|
priv struct Path {
  candidates : ReadOnlyArray[Candidate]
  ids : ReadOnlyArray[Int]
  nanoseconds : Int64
}

///|
/// Global minimum-cost cover in the declared topological order, not greedy
/// maximal fusion. Costs add because this plan executes its regions serially;
/// concurrent region timing requires a different cost model. Workspace is a
/// reusable per-region scratch arena, not a sum of persistent output storage.
pub fn select_regions(
  graph : Graph,
  candidates : ArrayView[Candidate],
  maximum_workspace_bytes~ : Int64,
  maximum_local_bytes~ : Int64,
) -> Selection raise FusionRegionError {
  if candidates.is_empty() ||
    maximum_workspace_bytes < 0L ||
    maximum_local_bytes < 0L {
    raise InvalidComparison
  }
  for index, candidate in candidates {
    if candidate.region.graph != graph ||
      candidate.cost.basis != candidates[0].cost.basis {
      raise InvalidComparison
    }
    for prior in 0..<index {
      if candidate.id == candidates[prior].id {
        raise InvalidComparison
      }
    }
  }
  let paths : Array[Path?] = Array::make(graph.node_count() + 1, None)
  paths[graph.node_count()] = Some({ candidates: [], ids: [], nanoseconds: 0L, })
  for reverse in 0..<graph.node_count() {
    let start = graph.node_count() - reverse - 1
    let best = for candidate in candidates; best = (None : Path?) {
      if candidate.region.start != start ||
        candidate.workspace_bytes > maximum_workspace_bytes ||
        candidate.local_bytes > maximum_local_bytes {
        continue best
      }
      guard paths[start + candidate.region.count] is Some(tail) else {
        continue best
      }
      if tail.nanoseconds > 9223372036854775807L - candidate.cost.nanoseconds {
        raise InvalidCost
      }
      let nanoseconds = candidate.cost.nanoseconds + tail.nanoseconds
      let ids = [candidate.id]
      ids.append(tail.ids[:])
      if best is Some(previous) &&
        (
          previous.nanoseconds < nanoseconds ||
          (
            previous.nanoseconds == nanoseconds &&
            !ids_less(ids, previous.ids[:])
          )
        ) {
        continue best
      }
      let path = [candidate]
      path.append(tail.candidates[:])
      continue Some({
          candidates: ReadOnlyArray::from_array(path),
          ids: ReadOnlyArray::from_array(ids),
          nanoseconds,
        })
    } nobreak {
      best
    }
    paths[start] = best
  }
  guard paths[0] is Some(best) else { raise NoFeasiblePartition }
  {
    partition: Partition::new(
      graph,
      best.candidates.map(candidate => candidate.region)[:],
    ),
    ids: best.ids,
    cost: { nanoseconds: best.nanoseconds, basis: candidates[0].cost.basis, },
  }
}

///|
/// An end-to-end observation of a whole partition is distinct from summing
/// individual region measurements (cache and launch interactions may differ).
pub struct PartitionAlternative {
  priv id : Int
  priv partition : Partition
  priv cost : Cost
  priv workspace_bytes : Int64
  priv local_bytes : Int64
  priv continuation_class : ContinuationClass
}

///|
pub fn PartitionAlternative::new(
  id~ : Int,
  partition : Partition,
  cost : Cost,
  workspace_bytes~ : Int64,
  local_bytes~ : Int64,
  continuation_class? : ContinuationClass = Materialized,
) -> PartitionAlternative raise FusionRegionError {
  if id < 0 ||
    workspace_bytes < 0L ||
    local_bytes < 0L ||
    !continuation_class.valid() {
    raise InvalidCost
  }
  { id, partition, cost, workspace_bytes, local_bytes, continuation_class, }
}

///|
pub fn select_partition(
  alternatives : ArrayView[PartitionAlternative],
  maximum_workspace_bytes~ : Int64,
  maximum_local_bytes~ : Int64,
  continuation_class? : ContinuationClass = Materialized,
) -> Int raise FusionRegionError {
  PartitionFrontier::new(alternatives)
  .select(continuation_class~, maximum_workspace_bytes~, maximum_local_bytes~)
  .implementation_id()
}

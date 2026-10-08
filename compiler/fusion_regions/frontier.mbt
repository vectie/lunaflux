///|
/// An existing materialized boundary or an owner-interned continuation. Scoped
/// IDs are local to one comparison, not model names or backend instructions.
pub(all) enum ContinuationClass {
  Materialized
  Scoped(Int)
} derive(Eq, Debug)

///|
pub extend ContinuationClass with Eq::{equal, not_equal}

///|
pub extend ContinuationClass with @debug.Debug::{to_repr}

///|
fn ContinuationClass::valid(self : ContinuationClass) -> Bool {
  match self {
    Materialized => true
    Scoped(id) => id >= 0
  }
}

///|
/// Nondominated complete implementations retained before the consumer and
/// resource envelope are resolved. No timing or resource value is invented.
///
/// A continuation class is a comparison-local identity supplied by the graph
/// owner. Equality promises the same live-value layouts, numerical policy,
/// ownership handoff, effects and cost-relevant continuation context (including
/// residency/cache and launch assumptions). Unknown context must use distinct
/// classes or whole-chain measurements. Materialized is the existing materialized
/// boundary. Different classes are never pruned against one another. This
/// identifier neither authenticates an ABI nor authorizes a layout conversion.
pub struct PartitionFrontier {
  priv alternatives : ReadOnlyArray[PartitionAlternative]
}

///|
/// Build an immutable frontier independent of enumeration order. Region-level
/// timings must not be passed as complete-partition observations. The caller
/// retains responsibility for workload/artifact measurement comparability.
pub fn PartitionFrontier::new(
  alternatives : ArrayView[PartitionAlternative],
) -> PartitionFrontier raise FusionRegionError {
  if alternatives.is_empty() {
    raise InvalidComparison
  }
  for index, candidate in alternatives {
    if candidate.partition.graph != alternatives[0].partition.graph ||
      candidate.cost.basis != alternatives[0].cost.basis {
      raise InvalidComparison
    }
    for prior in 0..<index {
      if candidate.id == alternatives[prior].id {
        raise InvalidComparison
      }
    }
  }
  let retained = alternatives.filter(candidate => {
    !alternatives.any(other => dominates(other, candidate))
  })
  retained.sort_by((a, b) => a.id.compare(b.id))
  { alternatives: ReadOnlyArray::from_array(retained), }
}

///|
fn dominates(left : PartitionAlternative, right : PartitionAlternative) -> Bool {
  left.id != right.id &&
  left.continuation_class == right.continuation_class &&
  left.workspace_bytes <= right.workspace_bytes &&
  left.local_bytes <= right.local_bytes &&
  (
    left.cost.nanoseconds < right.cost.nanoseconds ||
    (left.cost.nanoseconds == right.cost.nanoseconds && left.id < right.id)
  )
}

///|
/// Stable catalog order; exposing the retained alternatives allows offline
/// continuation composition without prematurely committing to one producer.
pub fn PartitionFrontier::alternatives(
  self : PartitionFrontier,
) -> ReadOnlyArray[PartitionAlternative] {
  self.alternatives
}

///|
/// Resolve one consumer-compatible class using final resource ceilings. The
/// frontier is reusable for another envelope. It does not sum persistent
/// storage or model concurrent regions; those remain separate planner inputs.
pub fn PartitionFrontier::select(
  self : PartitionFrontier,
  continuation_class~ : ContinuationClass,
  maximum_workspace_bytes~ : Int64,
  maximum_local_bytes~ : Int64,
) -> PartitionAlternative raise FusionRegionError {
  if !continuation_class.valid() ||
    maximum_workspace_bytes < 0L ||
    maximum_local_bytes < 0L {
    raise InvalidComparison
  }
  let best = for
    candidate in self.alternatives
    best = (None : PartitionAlternative?) {
    if candidate.continuation_class != continuation_class ||
      candidate.workspace_bytes > maximum_workspace_bytes ||
      candidate.local_bytes > maximum_local_bytes {
      continue best
    }
    continue match best {
        Some(previous) if previous.cost.nanoseconds < candidate.cost.nanoseconds ||
          (
            previous.cost.nanoseconds == candidate.cost.nanoseconds &&
            previous.id < candidate.id
          ) => best
        _ => Some(candidate)
      }
  } nobreak {
    best
  }
  match best {
    Some(candidate) => candidate
    None => raise NoFeasiblePartition
  }
}

///|
pub fn PartitionAlternative::implementation_id(
  self : PartitionAlternative,
) -> Int {
  self.id
}

///|
pub fn PartitionAlternative::continuation_class(
  self : PartitionAlternative,
) -> ContinuationClass {
  self.continuation_class
}

///|
pub fn PartitionAlternative::partition(
  self : PartitionAlternative,
) -> Partition {
  self.partition
}

///|
pub fn PartitionAlternative::cost(self : PartitionAlternative) -> Cost {
  self.cost
}

///|
pub fn PartitionAlternative::resources(
  self : PartitionAlternative,
) -> (Int64, Int64) {
  (self.workspace_bytes, self.local_bytes)
}

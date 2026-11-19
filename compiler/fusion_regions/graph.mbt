///|
/// Ordered effects retain their original order inside a region. A boundary
/// effect must remain a standalone region (for example an externally observed
/// synchronization). This vocabulary never authorizes deleting an effect.
pub(all) enum Effect {
  Pure
  Ordered
  Boundary
} derive(Eq, Debug)

///|
pub extend Effect with Eq::{equal, not_equal}

///|
pub extend Effect with @debug.Debug::{to_repr}

///|
pub(all) suberror FusionRegionError {
  InvalidGraph
  InvalidRegion
  MissingExport
  EffectBoundary
  InvalidPartition
  InvalidCost
  InvalidComparison
  NoFeasiblePartition
} derive(Eq, Debug)

///|
pub extend FusionRegionError with Eq::{equal, not_equal}

///|
pub extend FusionRegionError with @debug.Debug::{to_repr}

///|
/// One SSA value per node. Inputs outside this graph are not edges; the owner
/// binds those operands separately. Observable values must be materialized.
pub struct Node {
  priv inputs : ReadOnlyArray[Int32]
  priv observable : Bool
  priv effect : Effect
} derive(Eq)

///|
pub extend Node with Eq::{equal, not_equal}

///|
pub fn Node::new(
  inputs : ArrayView[Int32],
  observable~ : Bool,
  effect~ : Effect,
) -> Node {
  { inputs: ReadOnlyArray::from_array(inputs.to_owned()), observable, effect, }
}

///|
/// A topological operation order is part of the numerical/effect contract.
/// Partitioning never reorders nodes or reassociates arithmetic.
pub struct Graph {
  priv nodes : ReadOnlyArray[Node]
} derive(Eq)

///|
pub extend Graph with Eq::{equal, not_equal}

///|
pub fn Graph::new(nodes : ArrayView[Node]) -> Graph raise FusionRegionError {
  if nodes.is_empty() {
    raise InvalidGraph
  }
  for index, node in nodes {
    for input in node.inputs {
      if input < 0 || input >= index {
        raise InvalidGraph
      }
    }
  }
  { nodes: ReadOnlyArray::from_array(nodes.to_owned()), }
}

///|
pub fn Graph::node_count(self : Graph) -> Int32 {
  self.nodes.length()
}

///|
/// A supplied implementation of a contiguous topological region. Catalogs
/// provide implementations; this planner does not pretend arbitrary DAG
/// subgraphs already have executable fused kernels.
pub struct Region {
  priv graph : Graph
  priv start : Int32
  priv count : Int32
  priv exports : ReadOnlyArray[Int32]
}

///|
pub fn Region::new(
  graph : Graph,
  start~ : Int32,
  count~ : Int32,
  exports : ArrayView[Int32],
) -> Region raise FusionRegionError {
  if start < 0 || count <= 0 || start > graph.node_count() - count {
    raise InvalidRegion
  }
  let end = start + count
  for index, value in exports {
    if value < start ||
      value >= end ||
      (index > 0 && exports[index - 1] >= value) {
      raise InvalidRegion
    }
  }
  for index in start..<end {
    let node = graph.nodes[index]
    if node.effect == Boundary && count != 1 {
      raise EffectBoundary
    }
    let externally_used = graph.nodes[end:].any(consumer => {
      consumer.inputs.contains(index)
    })
    if (node.observable || externally_used) && !exports.contains(index) {
      raise MissingExport
    }
  }
  {
    graph,
    start,
    count,
    exports: ReadOnlyArray::from_array(exports.to_owned()),
  }
}

///|
pub fn Region::span(self : Region) -> (Int32, Int32) {
  (self.start, self.count)
}

///|
pub fn Region::exports(self : Region) -> ReadOnlyArray[Int32] {
  self.exports
}

///|
pub fn Region::effect_order(self : Region) -> ReadOnlyArray[Int32] {
  ReadOnlyArray::from_array(
    [
      for index in self.start..<(self.start + self.count) if self.graph.nodes[index].effect !=
      Pure => index
    ],
  )
}

///|
pub struct Partition {
  priv graph : Graph
  priv regions : ReadOnlyArray[Region]
}

///|
pub fn Partition::new(
  graph : Graph,
  regions : ArrayView[Region],
) -> Partition raise FusionRegionError {
  let end = for region in regions; end = 0 {
    if region.graph != graph || region.start != end {
      raise InvalidPartition
    }
    continue end + region.count
  } nobreak {
    end
  }
  if end != graph.node_count() {
    raise InvalidPartition
  }
  { graph, regions: ReadOnlyArray::from_array(regions.to_owned()), }
}

///|
pub fn Partition::regions(self : Partition) -> ReadOnlyArray[Region] {
  self.regions
}

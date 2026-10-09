///|
pub(all) enum NetworkTopologyRule {
  NodeIdentity
  DeviceOrdinal
  Memory
  Interface
  DuplicateNode
  TargetMismatch
} derive(Debug, Eq)

///|
/// Bounded error; node and interface labels are never diagnostic payloads.
pub(all) suberror NetworkTopologyError {
  Invalid(NetworkTopologyRule)
} derive(Debug, Eq)

///|
/// Deployment label only. This value performs no identity authentication.
pub struct NodeIdentity(String) derive(Debug, Eq)

///|
fn valid_label(value : StringView, maximum : Int) -> Bool {
  if value.is_empty() || value.length() > maximum {
    return false
  }
  for unit in value {
    match unit {
      'a'..='z' | 'A'..='Z' | '0'..='9' | '_' | '-' | '.' => ()
      _ => return false
    }
  }
  true
}

///|
pub fn NodeIdentity::new(
  value : StringView,
) -> NodeIdentity raise NetworkTopologyError {
  if !valid_label(value, 128) {
    raise Invalid(NodeIdentity)
  }
  NodeIdentity(value.to_owned())
}

///|
pub fn NodeIdentity::label(self : NodeIdentity) -> String {
  self.0
}

///|
/// One declared node-local device and RDMA interface. No handles or endpoints.
pub struct NetworkRank {
  priv node : NodeIdentity
  priv device_ordinal : Int
  priv target : @catalog.DeviceTarget
  priv physical_memory_bytes : Int64
  priv rdma_interface : String
} derive(Debug, Eq)

///|
pub fn NetworkRank::new(
  node~ : NodeIdentity,
  device_ordinal~ : Int,
  target~ : @catalog.DeviceTarget,
  physical_memory_bytes~ : Int64,
  rdma_interface~ : StringView,
) -> NetworkRank raise NetworkTopologyError {
  if device_ordinal < 0 || device_ordinal > 1023 {
    raise Invalid(DeviceOrdinal)
  }
  if physical_memory_bytes <= 0L {
    raise Invalid(Memory)
  }
  if !valid_label(rdma_interface, 63) {
    raise Invalid(Interface)
  }
  {
    node,
    device_ordinal,
    target,
    physical_memory_bytes,
    rdma_interface: rdma_interface.to_owned(),
  }
}

///|
pub fn NetworkRank::node(self : NetworkRank) -> NodeIdentity {
  self.node
}

///|
pub fn NetworkRank::device_ordinal(self : NetworkRank) -> Int {
  self.device_ordinal
}

///|
pub fn NetworkRank::target(self : NetworkRank) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn NetworkRank::physical_memory_bytes(self : NetworkRank) -> Int64 {
  self.physical_memory_bytes
}

///|
pub fn NetworkRank::rdma_interface(self : NetworkRank) -> String {
  self.rdma_interface
}

///|
/// Exact two-node RoCE declaration. Rank order is explicit and never sorted.
/// This is planning data, not proof of connectivity, NCCL readiness, or speed.
pub struct TwoNodeTopology {
  priv rank_zero : NetworkRank
  priv rank_one : NetworkRank
} derive(Debug, Eq)

///|
pub fn TwoNodeTopology::new(
  rank_zero : NetworkRank,
  rank_one : NetworkRank,
) -> TwoNodeTopology raise NetworkTopologyError {
  if rank_zero.node == rank_one.node {
    raise Invalid(DuplicateNode)
  }
  if rank_zero.target != rank_one.target {
    raise Invalid(TargetMismatch)
  }
  { rank_zero, rank_one, }
}

///|
pub fn TwoNodeTopology::rank(
  self : TwoNodeTopology,
  rank : Int,
) -> NetworkRank? {
  match rank {
    0 => Some(self.rank_zero)
    1 => Some(self.rank_one)
    _ => None
  }
}

///|
pub fn TwoNodeTopology::target(self : TwoNodeTopology) -> @catalog.DeviceTarget {
  self.rank_zero.target
}

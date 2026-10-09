///|
/// Projects a node-qualified declaration into a rank-local execution identity.
/// This is pure planning, not local peer-access or live device admission.
/// The remote startup owner must retain and authenticate the node identity.
pub fn TensorParallelRankDevice::from_network_topology(
  topology : @network_topology.TwoNodeTopology,
  rank~ : Int,
) -> TensorParallelRankDevice raise TensorParallelDevicePlanError {
  guard topology.rank(rank) is Some(device) else { raise invalid(Rank, rank) }
  {
    rank,
    world_size: 2,
    device_ordinal: device.device_ordinal(),
    target: device.target(),
  }
}

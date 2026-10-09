///|
pub(all) suberror NetworkFabricError {
  NodeMismatch
  TransportMismatch
  InterfaceMismatch
}

///|
fn validate_values(
  node : String,
  interface : String,
  observed_node : String?,
  network : String?,
  ib_disabled : String?,
  hca : String?,
) -> Unit raise NetworkFabricError {
  if observed_node != Some(node) {
    raise NodeMismatch
  }
  if network != Some("IB") || ib_disabled != Some("0") {
    raise TransportMismatch
  }
  if hca != Some("=" + interface) {
    raise InterfaceMismatch
  }
}

///|
/// Validate operator-pinned per-process settings before any NCCL communicator
/// opens. Never changes process environment or interprets this as link evidence.
/// One process owns one remote rank; environment changes after startup are invalid.
pub fn validate_process(
  rank : @network_topology.NetworkRank,
) -> Unit raise NetworkFabricError {
  validate_values(
    rank.node().label(),
    rank.rdma_interface(),
    @env.get_env_var("LUNAFLUX_NODE_ID"),
    @env.get_env_var("NCCL_NET"),
    @env.get_env_var("NCCL_IB_DISABLE"),
    @env.get_env_var("NCCL_IB_HCA"),
  )
}

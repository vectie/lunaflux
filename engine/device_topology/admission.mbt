///|
fn has_peer_access(
  peers : ReadOnlyArray[LocalPeerAccess],
  source : Int,
  destination : Int,
) -> Bool {
  for peer in peers {
    if peer.source_ordinal == source && peer.destination_ordinal == destination {
      return true
    }
  }
  false
}

///|
/// Joins an explicit same-host declaration with exact host observations.
pub fn admit_local_topology(
  declaration : LocalTopologyDeclaration,
  projection : LocalTopologyProbeProjection,
) -> AdmittedLocalTopology raise LocalTopologyError {
  if declaration.ranks.length() != projection.devices.length() {
    raise invalid(
      AdmissionRanks,
      CountMismatch,
      declaration.ranks.length(),
      projection.devices.length(),
    )
  }
  for index, declared in declaration.ranks {
    let observed = projection.devices[index]
    if declared.ordinal != observed.ordinal ||
      declared.name != observed.name ||
      declared.total_memory_bytes != observed.total_memory_bytes {
      raise invalid(AdmissionRanks, IdentityMismatch, index, index)
    }
    if declared.target != observed.target {
      raise invalid(AdmissionRanks, TargetMismatch, index, index)
    }
  }
  for lower in 0..<declaration.ranks.length() {
    for upper in (lower + 1)..<declaration.ranks.length() {
      let left = declaration.ranks[lower].ordinal
      let right = declaration.ranks[upper].ordinal
      if !has_peer_access(projection.peer_access, left, right) ||
        !has_peer_access(projection.peer_access, right, left) {
        raise invalid(AdmissionLinks, Unsupported, lower, upper)
      }
    }
  }
  {
    driver_version: projection.driver_version,
    ranks: declaration.ranks,
    target: declaration.ranks[0].target,
    link_count: declaration.links.length(),
  }
}

///|
/// Projects the physical local inventory and immediately applies the existing
/// exact declaration admission. It creates no context, worker, or collective.
pub fn admit_local_topology_from_inventory(
  declaration : LocalTopologyDeclaration,
) -> AdmittedLocalTopology raise LocalTopologyError {
  admit_local_topology(
    declaration,
    LocalTopologyProbeProjection::from_local_inventory(),
  )
}

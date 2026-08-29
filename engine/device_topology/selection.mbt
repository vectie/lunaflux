///|
/// Exact declaration and physical observation for one rank-ordered local
/// topology. It owns no device context, peer handle, or collective resource.
pub struct AdmittedRankOrderedLocalTopology {
  priv declaration : LocalTopologyDeclaration
  priv topology : AdmittedLocalTopology
}

///|
pub fn AdmittedRankOrderedLocalTopology::declaration(
  self : AdmittedRankOrderedLocalTopology,
) -> LocalTopologyDeclaration {
  self.declaration
}

///|
pub fn AdmittedRankOrderedLocalTopology::topology(
  self : AdmittedRankOrderedLocalTopology,
) -> AdmittedLocalTopology {
  self.topology
}

///|
fn rank_ordered_topology_from_projection(
  ordinals : ArrayView[Int],
  target : @catalog.DeviceTarget,
  projection : LocalTopologyProbeProjection,
) -> AdmittedRankOrderedLocalTopology raise LocalTopologyError {
  if ordinals.length() < MIN_LOCAL_RANKS || ordinals.length() > MAX_LOCAL_RANKS {
    raise invalid(DeclarationRanks, TooFew, ordinals.length(), MIN_LOCAL_RANKS)
  }
  let ranks : Array[LocalTopologyRank] = []
  let devices : Array[LocalProbeDevice] = []
  for rank, ordinal in ordinals {
    if ordinal < 0 ||
      ordinal > 1023 ||
      (rank > 0 && ordinal <= ordinals[rank - 1]) {
      raise invalid(DeclarationRanks, RankOrder, rank - 1, rank)
    }
    let mut selected : LocalProbeDevice? = None
    for observed in projection.devices {
      if observed.ordinal == ordinal {
        selected = Some(observed)
      }
    }
    guard selected is Some(observed) else {
      raise invalid(AdmissionRanks, IdentityMismatch, rank, rank)
    }
    if observed.target != target {
      raise invalid(AdmissionRanks, TargetMismatch, rank, rank)
    }
    ranks.push(
      LocalTopologyRank::new(
        ordinal~,
        name=observed.name,
        total_memory_bytes=observed.total_memory_bytes,
        target~,
      ),
    )
    devices.push(observed)
  }
  let links : Array[LocalTopologyLink] = []
  let peers : Array[LocalPeerAccess] = []
  for lower in 0..<ordinals.length() {
    for upper in (lower + 1)..<ordinals.length() {
      let left = ordinals[lower]
      let right = ordinals[upper]
      if !has_peer_access(projection.peer_access, left, right) ||
        !has_peer_access(projection.peer_access, right, left) {
        raise invalid(AdmissionLinks, Unsupported, lower, upper)
      }
      links.push(LocalTopologyLink::direct(lower_rank=lower, upper_rank=upper))
      peers.push(
        LocalPeerAccess::new(source_ordinal=left, destination_ordinal=right),
      )
      peers.push(
        LocalPeerAccess::new(source_ordinal=right, destination_ordinal=left),
      )
    }
  }
  let declaration = LocalTopologyDeclaration::new(ranks, links)
  let topology = admit_local_topology(
    declaration,
    LocalTopologyProbeProjection::from_observations(
      driver_version=projection.driver_version,
      devices,
      peers,
    ),
  )
  { declaration, topology }
}

///|
/// Probes the local inventory and admits exactly the declared rank order,
/// homogeneous target, and complete bidirectional peer mesh. This opens no
/// device context and creates no collective resource.
pub fn admit_rank_ordered_local_topology(
  ordinals : ArrayView[Int],
  target : @catalog.DeviceTarget,
) -> AdmittedRankOrderedLocalTopology raise LocalTopologyError {
  rank_ordered_topology_from_projection(
    ordinals,
    target,
    LocalTopologyProbeProjection::from_local_inventory(),
  )
}

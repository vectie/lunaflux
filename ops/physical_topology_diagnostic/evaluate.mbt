///|
fn topology_rejection(
  stage : PhysicalTopologyDiagnosticStage,
  error : @device_topology.LocalTopologyError,
) -> PhysicalTopologyDiagnosticRejection {
  match error {
    InvalidLocalTopology(area~, rule~, first~, second~) =>
      TopologyRejected(stage~, area~, rule~, first~, second~)
  }
}

///|
fn require_canonical_peer_matrix(
  devices : ArrayView[PhysicalTopologyDeviceObservation],
  peers : ArrayView[PhysicalPeerObservation],
) -> Unit raise PhysicalTopologyDiagnosticError {
  let required = devices.length() * (devices.length() - 1)
  if peers.length() != required {
    raise InvalidObservation
  }
  let mut cursor = 0
  for source_index, source in devices {
    for destination_index, destination in devices {
      if source_index != destination_index {
        let peer = peers[cursor]
        if peer.source_ordinal != source.ordinal ||
          peer.destination_ordinal != destination.ordinal {
          raise InvalidObservation
        }
        cursor += 1
      }
    }
  }
}

///|
fn collective_rejection(
  collective : PhysicalCollectiveObservation,
) -> PhysicalTopologyDiagnosticRejection? {
  match collective.availability {
    CollectiveRuntimeAvailable =>
      if collective.admitted_version_code == 0 {
        Some(CollectiveVersionRejected)
      } else {
        None
      }
    unavailable => Some(CollectiveUnavailable(unavailable))
  }
}

///|
fn rejected_evidence(
  driver_version : Int,
  devices : ArrayView[PhysicalTopologyDeviceObservation],
  peers : ArrayView[PhysicalPeerObservation],
  peer_query_complete : Bool,
  collective : PhysicalCollectiveObservation,
  rejection : PhysicalTopologyDiagnosticRejection,
  admitted_topology? : @device_topology.AdmittedLocalTopology,
) -> PhysicalTopologyDiagnosticEvidence {
  {
    driver_version,
    devices: ReadOnlyArray::from_array(devices),
    peers: ReadOnlyArray::from_array(peers),
    peer_query_complete,
    collective,
    admitted_topology,
    rejection: Some(rejection),
  }
}

///|
fn evaluate_topology(
  driver_version : Int,
  devices : ArrayView[PhysicalTopologyDeviceObservation],
  peers : ArrayView[PhysicalPeerObservation],
) -> Result[
  @device_topology.AdmittedLocalTopology,
  PhysicalTopologyDiagnosticRejection,
] {
  let projected_devices : Array[@device_topology.LocalProbeDevice] = []
  let ranks : Array[@device_topology.LocalTopologyRank] = []
  for index, device in devices {
    let target = @catalog.DeviceTarget::from_capability(device.capability) catch {
      _ =>
        return Err(
          TopologyRejected(
            stage=TopologyProjection,
            area=ProbeDevices,
            rule=TargetMismatch,
            first=index,
            second=index,
          ),
        )
    }
    let projected = @device_topology.LocalProbeDevice::new(
      ordinal=device.ordinal,
      name=device.name,
      total_memory_bytes=device.capability.total_memory_bytes(),
      target~,
    ) catch {
      error => return Err(topology_rejection(TopologyProjection, error))
    }
    let rank = @device_topology.LocalTopologyRank::new(
      ordinal=device.ordinal,
      name=device.name,
      total_memory_bytes=device.capability.total_memory_bytes(),
      target~,
    ) catch {
      error => return Err(topology_rejection(TopologyDeclaration, error))
    }
    projected_devices.push(projected)
    ranks.push(rank)
  }
  let accessible : Array[@device_topology.LocalPeerAccess] = []
  for peer in peers {
    if peer.accessible {
      let edge = @device_topology.LocalPeerAccess::new(
        source_ordinal=peer.source_ordinal,
        destination_ordinal=peer.destination_ordinal,
      ) catch {
        error => return Err(topology_rejection(TopologyProjection, error))
      }
      accessible.push(edge)
    }
  }
  let projection = @device_topology.LocalTopologyProbeProjection::from_observations(
    driver_version~,
    projected_devices,
    accessible,
  ) catch {
    error => return Err(topology_rejection(TopologyProjection, error))
  }
  let links : Array[@device_topology.LocalTopologyLink] = []
  for lower in 0..<ranks.length() {
    for upper in (lower + 1)..<ranks.length() {
      let link = @device_topology.LocalTopologyLink::direct(
        lower_rank=lower,
        upper_rank=upper,
      ) catch {
        error => return Err(topology_rejection(TopologyDeclaration, error))
      }
      links.push(link)
    }
  }
  let declaration = @device_topology.LocalTopologyDeclaration::new(ranks, links) catch {
    error => return Err(topology_rejection(TopologyDeclaration, error))
  }
  let admission = @device_topology.admit_local_topology(declaration, projection) catch {
    error => return Err(topology_rejection(TopologyAdmission, error))
  }
  Ok(admission)
}

///|
/// Evaluates one complete canonical host observation matrix through the exact
/// production topology admission. This pure seam creates no native authority.
pub fn evaluate_observations(
  driver_version~ : Int,
  devices : ArrayView[PhysicalTopologyDeviceObservation],
  peers : ArrayView[PhysicalPeerObservation],
  collective~ : PhysicalCollectiveObservation,
) -> PhysicalTopologyDiagnosticEvidence raise PhysicalTopologyDiagnosticError {
  if driver_version <= 0 {
    raise InvalidObservation
  }
  require_canonical_peer_matrix(devices, peers)
  match evaluate_topology(driver_version, devices, peers) {
    Err(rejection) =>
      rejected_evidence(
        driver_version, devices, peers, true, collective, rejection,
      )
    Ok(admitted) =>
      match collective_rejection(collective) {
        Some(rejection) =>
          rejected_evidence(
            driver_version,
            devices,
            peers,
            true,
            collective,
            rejection,
            admitted_topology=admitted,
          )
        None =>
          {
            driver_version,
            devices: ReadOnlyArray::from_array(devices),
            peers: ReadOnlyArray::from_array(peers),
            peer_query_complete: true,
            collective,
            admitted_topology: Some(admitted),
            rejection: None,
          }
      }
  }
}

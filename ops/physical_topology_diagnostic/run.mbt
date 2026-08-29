///|
fn observe_collective_runtime() -> PhysicalCollectiveObservation {
  let inventory = @device.probe_tensor_collective_runtime()
  let admitted_version = match inventory.availability() {
    CollectiveRuntimeAvailable => {
      let runtime = @device.admit_tensor_collective_runtime(
        minimum_version_code=MINIMUM_COLLECTIVE_RUNTIME_VERSION,
        maximum_version_code=MAXIMUM_COLLECTIVE_RUNTIME_VERSION,
      ) catch {
        _ =>
          return {
            availability: inventory.availability(),
            version_code: inventory.version_code(),
            admitted_version_code: 0,
          }
      }
      runtime.collective_runtime_version_code()
    }
    _ => 0
  }
  {
    availability: inventory.availability(),
    version_code: inventory.version_code(),
    admitted_version_code: admitted_version,
  }
}

///|
fn unavailable_device_evidence(
  reason : @device.UnavailableReason,
  collective : PhysicalCollectiveObservation,
) -> PhysicalTopologyDiagnosticEvidence {
  rejected_evidence(0, [], [], false, collective, DeviceUnavailable(reason))
}

///|
fn failed_peer_evidence(
  driver_version : Int,
  devices : Array[PhysicalTopologyDeviceObservation],
  peers : Array[PhysicalPeerObservation],
  source_rank : Int,
  destination_rank : Int,
  collective : PhysicalCollectiveObservation,
) -> PhysicalTopologyDiagnosticEvidence {
  rejected_evidence(
    driver_version,
    devices,
    peers,
    false,
    collective,
    PeerQueryFailed(source_rank~, destination_rank~),
  )
}

///|
/// Runs the product-owned physical Phase-7 diagnostic. Expected unsupported
/// hosts return immutable rejection evidence instead of raising or degrading.
/// No context, communicator, rank process, peer enable, or allocation is
/// created.
pub fn run_local() -> PhysicalTopologyDiagnosticEvidence {
  let collective = observe_collective_runtime()
  let (driver_version, device_infos) = match @device.probe() {
    Available(driver_version~, devices~) => (driver_version, devices)
    Unavailable(reason) =>
      return unavailable_device_evidence(reason, collective)
  }
  let devices : Array[PhysicalTopologyDeviceObservation] = []
  for info in device_infos {
    let observation = PhysicalTopologyDeviceObservation::new(
      ordinal=info.ordinal(),
      name=info.name(),
      capability=info.capability(),
    ) catch {
      _ => return unavailable_device_evidence(InventoryFailed, collective)
    }
    devices.push(observation)
  }
  let peers : Array[PhysicalPeerObservation] = []
  for source_rank, source in devices {
    for destination_rank, destination in devices {
      if source_rank != destination_rank {
        let accessible = @device.can_access_peer(
          source.ordinal,
          destination.ordinal,
        ) catch {
          _ =>
            return failed_peer_evidence(
              driver_version, devices, peers, source_rank, destination_rank, collective,
            )
        }
        peers.push({
          source_ordinal: source.ordinal,
          destination_ordinal: destination.ordinal,
          accessible,
        })
      }
    }
  }
  evaluate_observations(driver_version~, devices, peers, collective~) catch {
    _ => unavailable_device_evidence(InventoryFailed, collective)
  }
}

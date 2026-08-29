///|
pub fn LocalPeerAccess::new(
  source_ordinal~ : Int,
  destination_ordinal~ : Int,
) -> LocalPeerAccess raise LocalTopologyError {
  if source_ordinal < 0 ||
    destination_ordinal < 0 ||
    source_ordinal == destination_ordinal {
    raise invalid(ProbeLinks, Unsupported, source_ordinal, destination_ordinal)
  }
  { source_ordinal, destination_ordinal }
}

///|
fn device_index_for_ordinal(
  devices : ReadOnlyArray[LocalProbeDevice],
  ordinal : Int,
) -> Int {
  for index, device in devices {
    if device.ordinal == ordinal {
      return index
    }
  }
  -1
}

///|
fn validate_probe_devices(
  devices : ReadOnlyArray[LocalProbeDevice],
) -> Unit raise LocalTopologyError {
  if devices.length() < MIN_LOCAL_RANKS {
    raise invalid(ProbeDevices, TooFew, devices.length(), MIN_LOCAL_RANKS)
  }
  if devices.length() > MAX_LOCAL_RANKS {
    raise invalid(ProbeDevices, TooMany, devices.length(), MAX_LOCAL_RANKS)
  }
  for index, device in devices {
    if device.ordinal < 0 ||
      device.name.is_empty() ||
      device.name.length() > MAX_DEVICE_NAME_UNITS ||
      device.total_memory_bytes <= 0L {
      raise invalid(ProbeDevices, IdentityMismatch, index, index)
    }
    if index > 0 && device.ordinal <= devices[index - 1].ordinal {
      raise invalid(
        ProbeDevices,
        if device.ordinal == devices[index - 1].ordinal {
          Duplicate
        } else {
          RankOrder
        },
        index - 1,
        index,
      )
    }
  }
}

///|
fn validate_probe_links(
  devices : ReadOnlyArray[LocalProbeDevice],
  peer_access : ArrayView[LocalPeerAccess],
) -> Unit raise LocalTopologyError {
  let maximum = devices.length() * (devices.length() - 1)
  if peer_access.length() > maximum {
    raise invalid(ProbeLinks, TooMany, peer_access.length(), maximum)
  }
  let mut prior_source = -1
  let mut prior_destination = -1
  for index, peer in peer_access {
    if device_index_for_ordinal(devices, peer.source_ordinal) < 0 ||
      device_index_for_ordinal(devices, peer.destination_ordinal) < 0 {
      raise invalid(ProbeLinks, Unsupported, index, index)
    }
    if index > 0 &&
      (
        peer.source_ordinal < prior_source ||
        (
          peer.source_ordinal == prior_source &&
          peer.destination_ordinal <= prior_destination
        )
      ) {
      raise invalid(
        ProbeLinks,
        if peer.source_ordinal == prior_source &&
          peer.destination_ordinal == prior_destination {
          Duplicate
        } else {
          RankOrder
        },
        index - 1,
        index,
      )
    }
    prior_source = peer.source_ordinal
    prior_destination = peer.destination_ordinal
  }
}

///|
pub fn LocalProbeDevice::new(
  ordinal~ : Int,
  name~ : StringView,
  total_memory_bytes~ : Int64,
  target~ : @catalog.DeviceTarget,
) -> LocalProbeDevice raise LocalTopologyError {
  if ordinal < 0 || ordinal > 1023 {
    raise invalid(ProbeDevices, InvalidOrdinal, 0, 0)
  }
  if name.is_empty() || name.length() > MAX_DEVICE_NAME_UNITS {
    raise invalid(ProbeDevices, InvalidName, 0, 0)
  }
  if total_memory_bytes <= 0L {
    raise invalid(ProbeDevices, InvalidMemory, 0, 0)
  }
  { ordinal, name: name.to_owned(), total_memory_bytes, target }
}

///|
/// Constructs the pure host-test seam from already bounded observations.
pub fn LocalTopologyProbeProjection::from_observations(
  driver_version~ : Int,
  devices : ArrayView[LocalProbeDevice],
  peer_access : ArrayView[LocalPeerAccess],
) -> LocalTopologyProbeProjection raise LocalTopologyError {
  if driver_version <= 0 {
    raise invalid(ProbeDevices, ProbeUnavailable, 0, 0)
  }
  let owned_devices = ReadOnlyArray::from_array(devices)
  validate_probe_devices(owned_devices)
  validate_probe_links(owned_devices, peer_access)
  {
    driver_version,
    devices: owned_devices,
    peer_access: ReadOnlyArray::from_array(peer_access),
  }
}

///|
/// Projects bounded host observations without opening a context or granting
/// device/collective authority. Pure tests and offline inspectors may supply
/// already admitted directed peer observations; production uses the physical
/// inventory constructor below.
pub fn LocalTopologyProbeProjection::from_probe(
  probe : @device.ProbeResult,
  peer_access : ArrayView[LocalPeerAccess],
) -> LocalTopologyProbeProjection raise LocalTopologyError {
  guard probe is Available(driver_version~, devices~) else {
    raise invalid(ProbeDevices, ProbeUnavailable, 0, 0)
  }
  let projected : Array[LocalProbeDevice] = []
  for device in devices {
    let target = @catalog.DeviceTarget::from_capability(device.capability()) catch {
      _ =>
        raise invalid(
          ProbeDevices,
          TargetMismatch,
          projected.length(),
          projected.length(),
        )
    }
    projected.push(
      LocalProbeDevice::new(
        ordinal=device.ordinal(),
        name=device.name(),
        total_memory_bytes=device.capability().total_memory_bytes(),
        target~,
      ),
    )
  }
  LocalTopologyProbeProjection::from_observations(
    driver_version~,
    projected,
    peer_access,
  )
}

///|
fn collect_peer_access(
  devices : ReadOnlyArray[LocalProbeDevice],
  query : (Int, Int) -> Bool raise @device.DeviceError,
) -> Array[LocalPeerAccess] raise LocalTopologyError {
  let peer_access : Array[LocalPeerAccess] = []
  for source_index, source in devices {
    for destination_index, destination in devices {
      if source_index != destination_index {
        let available = query(source.ordinal, destination.ordinal) catch {
          _ =>
            raise invalid(
              ProbeLinks,
              ProbeUnavailable,
              source_index,
              destination_index,
            )
        }
        if available {
          peer_access.push(
            LocalPeerAccess::new(
              source_ordinal=source.ordinal,
              destination_ordinal=destination.ordinal,
            ),
          )
        }
      }
    }
  }
  peer_access
}

///|
/// Performs the production authority-free inventory projection. Every ordered
/// pair is queried exactly once in stable source-rank then destination-rank
/// order; false edges remain absent so the existing admission rejects an
/// unsupported initial full mesh.
pub fn LocalTopologyProbeProjection::from_local_inventory() -> LocalTopologyProbeProjection raise LocalTopologyError {
  let probe = @device.probe()
  let bounded = LocalTopologyProbeProjection::from_probe(probe, [])
  let peer_access = collect_peer_access(bounded.devices, (source, destination) => {
    @device.can_access_peer(source, destination)
  })
  LocalTopologyProbeProjection::from_observations(
    driver_version=bounded.driver_version,
    bounded.devices,
    peer_access,
  )
}

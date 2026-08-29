///|
fn diagnostic_device(
  ordinal : Int,
  name : String,
  major : Int,
  minor : Int,
  memory : Int64,
  bf16 : Bool,
) -> PhysicalTopologyDeviceObservation raise {
  @physical_topology_diagnostic.PhysicalTopologyDeviceObservation::new(
    ordinal~,
    name~,
    capability=@device.DeviceCapability::from_observed_facts(
      compute_major=major,
      compute_minor=minor,
      total_memory_bytes=memory,
      supports_bf16=bf16,
      supports_cublas_lt=true,
    ),
  )
}

///|
fn diagnostic_collective(
  availability : @device.TensorCollectiveRuntimeAvailability,
  version : Int,
  admitted : Int,
) -> PhysicalCollectiveObservation raise {
  @physical_topology_diagnostic.PhysicalCollectiveObservation::new(
    availability~,
    version_code=version,
    admitted_version_code=admitted,
  )
}

///|
fn diagnostic_peers(
  left : Int,
  right : Int,
  forward : Bool,
  reverse : Bool,
) -> Array[PhysicalPeerObservation] raise {
  [
    @physical_topology_diagnostic.PhysicalPeerObservation::new(
      source_ordinal=left,
      destination_ordinal=right,
      accessible=forward,
    ),
    @physical_topology_diagnostic.PhysicalPeerObservation::new(
      source_ordinal=right,
      destination_ordinal=left,
      accessible=reverse,
    ),
  ]
}

///|
test "mixed sm120 sm75 no-peer host is an exact successful rejection" {
  let devices = [
    diagnostic_device(0, "sm120", 12, 0, 17179869184L, true),
    diagnostic_device(1, "sm75", 7, 5, 8589934592L, false),
  ]
  let evidence = @physical_topology_diagnostic.evaluate_observations(
    driver_version=13010,
    devices,
    diagnostic_peers(0, 1, false, false),
    collective=diagnostic_collective(CollectiveRuntimeLibraryMissing, 0, 0),
  )
  assert_true(evidence.status() == PhysicalTopologyRejected)
  assert_true(evidence.admitted_topology() is None)
  let rendered = @physical_topology_diagnostic.render(evidence)
  assert_eq(
    rendered,
    (
      #|schema: lunaflux.phase7.physical_topology.v1
      #|outcome: rejected
      #|driver.version: 13010
      #|device.count: 2
      #|device.0.ordinal: 0
      #|device.0.name_utf8_hex: 736d313230
      #|device.0.compute_major: 12
      #|device.0.compute_minor: 0
      #|device.0.total_memory_bytes: 17179869184
      #|device.0.supports_bf16: true
      #|device.0.supports_cublas_lt: true
      #|device.1.ordinal: 1
      #|device.1.name_utf8_hex: 736d3735
      #|device.1.compute_major: 7
      #|device.1.compute_minor: 5
      #|device.1.total_memory_bytes: 8589934592
      #|device.1.supports_bf16: false
      #|device.1.supports_cublas_lt: true
      #|peer.query_complete: true
      #|peer.count: 2
      #|peer.0.source_ordinal: 0
      #|peer.0.destination_ordinal: 1
      #|peer.0.accessible: false
      #|peer.1.source_ordinal: 1
      #|peer.1.destination_ordinal: 0
      #|peer.1.accessible: false
      #|collective.required_minimum_version: 21403
      #|collective.required_maximum_version: 29999
      #|collective.availability: library_missing
      #|collective.observed_version: 0
      #|collective.admitted_version: 0
      #|topology.admitted: false
      #|rejection.stage: topology_declaration
      #|rejection.area: declaration_ranks
      #|rejection.reason: heterogeneous_target
      #|rejection.first: 0
      #|rejection.second: 1
      #|authority.context_opened: false
      #|authority.device_allocation_created: false
      #|authority.communicator_created: false
      #|authority.rank_process_spawned: false
      #|
    ),
  )
}

///|
test "homogeneous full mesh and admitted collective runtime is supported" {
  let devices = [
    diagnostic_device(2, "left", 9, 0, 17179869184L, true),
    diagnostic_device(5, "right", 9, 0, 17179869184L, true),
  ]
  let evidence = @physical_topology_diagnostic.evaluate_observations(
    driver_version=13010,
    devices,
    diagnostic_peers(2, 5, true, true),
    collective=diagnostic_collective(CollectiveRuntimeAvailable, 21403, 21403),
  )
  assert_true(evidence.status() == PhysicalTopologySupported)
  assert_true(evidence.admitted_topology() is Some(_))
  assert_true(evidence.rejection() is None)
}

///|
test "missing collective library rejects otherwise admitted topology" {
  let devices = [
    diagnostic_device(2, "left", 9, 0, 17179869184L, true),
    diagnostic_device(5, "right", 9, 0, 17179869184L, true),
  ]
  let evidence = @physical_topology_diagnostic.evaluate_observations(
    driver_version=13010,
    devices,
    diagnostic_peers(2, 5, true, true),
    collective=diagnostic_collective(CollectiveRuntimeLibraryMissing, 0, 0),
  )
  assert_true(evidence.admitted_topology() is Some(_))
  assert_true(
    evidence.rejection()
    is Some(CollectiveUnavailable(CollectiveRuntimeLibraryMissing)),
  )
}

///|
test "one-way peer and unadmitted version fail closed without degraded mode" {
  let devices = [
    diagnostic_device(2, "left", 9, 0, 17179869184L, true),
    diagnostic_device(5, "right", 9, 0, 17179869184L, true),
  ]
  let one_way = @physical_topology_diagnostic.evaluate_observations(
    driver_version=13010,
    devices,
    diagnostic_peers(2, 5, true, false),
    collective=diagnostic_collective(CollectiveRuntimeAvailable, 21403, 21403),
  )
  assert_true(
    one_way.rejection()
    is Some(
      TopologyRejected(
        stage=TopologyAdmission,
        area=AdmissionLinks,
        rule=Unsupported,
        first=0,
        second=1
      )
    ),
  )
  let version = @physical_topology_diagnostic.evaluate_observations(
    driver_version=13010,
    devices,
    diagnostic_peers(2, 5, true, true),
    collective=diagnostic_collective(CollectiveRuntimeAvailable, 21404, 0),
  )
  assert_true(version.rejection() is Some(CollectiveVersionRejected))
}

///|
test "hostile incomplete or reordered directed matrices are invalid evidence" {
  let devices = [
    diagnostic_device(2, "left", 9, 0, 1L, true),
    diagnostic_device(5, "right", 9, 0, 1L, true),
  ]
  let collective = diagnostic_collective(CollectiveRuntimeLibraryMissing, 0, 0)
  let hostile = [
    [diagnostic_peers(2, 5, true, true)[0]],
    [
      diagnostic_peers(2, 5, true, true)[1],
      diagnostic_peers(2, 5, true, true)[0],
    ],
  ]
  for peers in hostile {
    try
      @physical_topology_diagnostic.evaluate_observations(
        driver_version=13010,
        devices,
        peers,
        collective~,
      )
    catch {
      InvalidObservation => ()
    } noraise {
      _ => fail("hostile directed matrix was admitted")
    }
  }
}

///|
test "hostile collective observations cannot forge admission" {
  let cases = [
    (@device.CollectiveRuntimeAvailable, 0, 0),
    (CollectiveRuntimeAvailable, 21403, 21404),
    (CollectiveRuntimeLibraryMissing, 21403, 0),
    (CollectiveRuntimeLibraryMissing, 0, 21403),
    (CollectiveRuntimeAbiIncomplete, 21403, 21403),
  ]
  for test_case in cases {
    let (availability, version_code, admitted_version_code) = test_case
    try
      @physical_topology_diagnostic.PhysicalCollectiveObservation::new(
        availability~,
        version_code~,
        admitted_version_code~,
      )
    catch {
      InvalidObservation => ()
    } noraise {
      _ => fail("hostile collective observation was constructed")
    }
  }
}

///|
test "device names cannot inject evidence fields" {
  let devices = [
    diagnostic_device(2, "left\noutcome: supported", 9, 0, 1L, true),
    diagnostic_device(5, "right", 9, 0, 1L, true),
  ]
  let evidence = @physical_topology_diagnostic.evaluate_observations(
    driver_version=13010,
    devices,
    diagnostic_peers(2, 5, true, true),
    collective=diagnostic_collective(CollectiveRuntimeAvailable, 21403, 21403),
  )
  let rendered = @physical_topology_diagnostic.render(evidence)
  assert_true(
    rendered.contains(
      "device.0.name_utf8_hex: 6c6566740a6f7574636f6d653a20737570706f72746564",
    ),
  )
  assert_false(rendered.contains("left\noutcome: supported"))
}

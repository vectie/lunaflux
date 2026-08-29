///|
fn wb_topology_device(
  ordinal : Int,
  name : String,
) -> LocalProbeDevice raise LocalTopologyError {
  LocalProbeDevice::new(
    ordinal~,
    name~,
    total_memory_bytes=1L,
    target=try! @catalog.DeviceTarget::new(
      compute_major=8,
      compute_minor=0,
      supports_bf16=true,
      supports_cublas_lt=true,
    ),
  )
}

///|
test "physical pair projection is canonical directed and exact" {
  let devices = ReadOnlyArray::from_array([
    wb_topology_device(2, "two"),
    wb_topology_device(5, "five"),
    wb_topology_device(9, "nine"),
  ])
  let observed : Array[(Int, Int)] = []
  let peers = collect_peer_access(devices, (source, destination) => {
    observed.push((source, destination))
    source != 5 || destination != 2
  })
  assert_eq(observed, [(2, 5), (2, 9), (5, 2), (5, 9), (9, 2), (9, 5)])
  assert_eq(peers.length(), 5)
  assert_eq(peers[0].source_ordinal, 2)
  assert_eq(peers[0].destination_ordinal, 5)
  assert_eq(peers[4].source_ordinal, 9)
  assert_eq(peers[4].destination_ordinal, 5)
}

///|
test "physical pair query failure publishes no partial projection" {
  let devices = ReadOnlyArray::from_array([
    wb_topology_device(2, "two"),
    wb_topology_device(5, "five"),
  ])
  try
    collect_peer_access(devices, (source, destination) => {
      if source == 5 && destination == 2 {
        raise Unavailable(DevicePeerAccess)
      }
      true
    })
  catch {
    InvalidLocalTopology(
      area=ProbeLinks,
      rule=ProbeUnavailable,
      first=1,
      second=0
    ) => ()
    error => fail("unexpected peer query rejection: \{Repr(error)}")
  } noraise {
    _ => fail("failed directed query published a partial projection")
  }
}

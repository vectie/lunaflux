///|
fn topology_target(
  minor? : Int = 0,
) -> @catalog.DeviceTarget raise @catalog.CatalogError {
  @catalog.DeviceTarget::new(
    compute_major=8,
    compute_minor=minor,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
}

///|
fn topology_rank(
  ordinal : Int,
  name : String,
  memory : Int64,
  target : @catalog.DeviceTarget,
) -> LocalTopologyRank raise LocalTopologyError {
  @device_topology.LocalTopologyRank::new(
    ordinal~,
    name~,
    total_memory_bytes=memory,
    target~,
  )
}

///|
fn topology_device(
  ordinal : Int,
  name : String,
  memory : Int64,
  minor? : Int = 0,
) -> LocalProbeDevice raise LocalTopologyError {
  @device_topology.LocalProbeDevice::new(
    ordinal~,
    name~,
    total_memory_bytes=memory,
    target=try! topology_target(minor~),
  )
}

///|
fn two_rank_declaration(
  target : @catalog.DeviceTarget,
) -> LocalTopologyDeclaration raise LocalTopologyError {
  @device_topology.LocalTopologyDeclaration::new(
    [
      topology_rank(2, "rank-two", 17179869184L, target),
      topology_rank(5, "rank-five", 17179869184L, target),
    ],
    [@device_topology.LocalTopologyLink::direct(lower_rank=0, upper_rank=1)],
  )
}

///|
fn two_rank_projection(
  second_name? : String = "rank-five",
  second_minor? : Int = 0,
  reverse_peer? : Bool = true,
) -> LocalTopologyProbeProjection raise LocalTopologyError {
  let peers = [
    @device_topology.LocalPeerAccess::new(
      source_ordinal=2,
      destination_ordinal=5,
    ),
  ]
  if reverse_peer {
    peers.push(
      @device_topology.LocalPeerAccess::new(
        source_ordinal=5,
        destination_ordinal=2,
      ),
    )
  }
  @device_topology.LocalTopologyProbeProjection::from_observations(
    driver_version=12080,
    [
      topology_device(2, "rank-two", 17179869184L),
      topology_device(5, second_name, 17179869184L, minor=second_minor),
    ],
    peers,
  )
}

///|
test "exact non-contiguous local rank group admits deterministically" {
  let target = topology_target()
  let declaration = two_rank_declaration(target)
  let projection = two_rank_projection()
  let admitted = @device_topology.admit_local_topology(declaration, projection)
  assert_eq(declaration.rank_count(), 2)
  assert_eq(declaration.link_count(), 1)
  assert_eq(projection.device_count(), 2)
  assert_eq(projection.peer_access_count(), 2)
  assert_eq(admitted.driver_version(), 12080)
  assert_eq(admitted.world_size(), 2)
  assert_eq(admitted.link_count(), 1)
  assert_eq(admitted.target(), target)
  guard admitted.rank(0) is Some(first) && admitted.rank(1) is Some(second) else {
    fail("admitted rank order was lost")
  }
  assert_eq(first.ordinal(), 2)
  assert_eq(first.name(), "rank-two")
  assert_eq(second.ordinal(), 5)
  assert_eq(second.total_memory_bytes(), 17179869184L)
}

///|
test "declaration rejects reordered duplicate and heterogeneous ranks" {
  let target = topology_target()
  let other = topology_target(minor=6)
  let cases = [
    [topology_rank(5, "five", 1L, target), topology_rank(2, "two", 1L, target)],
    [
      topology_rank(2, "two", 1L, target),
      topology_rank(2, "duplicate", 1L, target),
    ],
    [topology_rank(2, "two", 1L, target), topology_rank(5, "five", 1L, other)],
  ]
  for ranks in cases {
    try
      @device_topology.LocalTopologyDeclaration::new(ranks, [
        @device_topology.LocalTopologyLink::direct(lower_rank=0, upper_rank=1),
      ])
    catch {
      InvalidLocalTopology(area=DeclarationRanks, ..) => ()
      _ => fail("unexpected invalid-rank rejection")
    } noraise {
      _ => fail("invalid rank declaration was admitted")
    }
  }
}

///|
test "rank and projection bounds fail before topology publication" {
  let target = topology_target()
  try
    @device_topology.LocalTopologyDeclaration::new(
      [topology_rank(0, "only", 1L, target)],
      [],
    )
  catch {
    InvalidLocalTopology(area=DeclarationRanks, rule=TooFew, ..) => ()
    _ => fail("unexpected minimum-rank rejection")
  } noraise {
    _ => fail("single-rank multi-device topology was admitted")
  }
  let ranks : Array[LocalTopologyRank] = []
  for ordinal in 0..<17 {
    ranks.push(topology_rank(ordinal, "rank", 1L, target))
  }
  try @device_topology.LocalTopologyDeclaration::new(ranks, []) catch {
    InvalidLocalTopology(area=DeclarationRanks, rule=TooMany, ..) => ()
    _ => fail("unexpected maximum-rank rejection")
  } noraise {
    _ => fail("oversized topology was admitted")
  }
  try
    @device_topology.LocalTopologyProbeProjection::from_observations(
      driver_version=12080,
      [topology_device(2, "two", 1L), topology_device(2, "duplicate", 1L)],
      [],
    )
  catch {
    InvalidLocalTopology(area=ProbeDevices, rule=Duplicate, ..) => ()
    _ => fail("unexpected duplicate-probe rejection")
  } noraise {
    _ => fail("duplicate probed ordinal was admitted")
  }
}

///|
test "initial supported topology rejects sparse reversed and unordered links" {
  let target = topology_target()
  let ranks = [
    topology_rank(2, "two", 1L, target),
    topology_rank(5, "five", 1L, target),
    topology_rank(9, "nine", 1L, target),
  ]
  try
    @device_topology.LocalTopologyDeclaration::new(ranks, [
      @device_topology.LocalTopologyLink::direct(lower_rank=0, upper_rank=1),
      @device_topology.LocalTopologyLink::direct(lower_rank=0, upper_rank=2),
    ])
  catch {
    InvalidLocalTopology(area=DeclarationLinks, rule=Incomplete, ..) => ()
    _ => fail("unexpected sparse-link rejection")
  } noraise {
    _ => fail("sparse topology was admitted")
  }
  try
    @device_topology.LocalTopologyLink::direct(lower_rank=1, upper_rank=0)
  catch {
    InvalidLocalTopology(area=DeclarationLinks, rule=Unsupported, ..) => ()
    _ => fail("unexpected reversed-link rejection")
  } noraise {
    _ => fail("reversed link was admitted")
  }
  try
    @device_topology.LocalTopologyDeclaration::new(ranks, [
      @device_topology.LocalTopologyLink::direct(lower_rank=0, upper_rank=2),
      @device_topology.LocalTopologyLink::direct(lower_rank=0, upper_rank=1),
      @device_topology.LocalTopologyLink::direct(lower_rank=1, upper_rank=2),
    ])
  catch {
    InvalidLocalTopology(area=DeclarationLinks, rule=Unsupported, ..) => ()
    _ => fail("unexpected unordered-link rejection")
  } noraise {
    _ => fail("unordered topology was admitted")
  }
}

///|
test "probe projection rejects unavailable duplicate and ambient links" {
  try
    @device_topology.LocalTopologyProbeProjection::from_probe(
      Unavailable(UnsupportedPlatform),
      [],
    )
  catch {
    InvalidLocalTopology(area=ProbeDevices, rule=ProbeUnavailable, ..) => ()
    _ => fail("unexpected unavailable-probe rejection")
  } noraise {
    _ => fail("unavailable probe was projected")
  }
  let devices = [topology_device(2, "two", 1L), topology_device(5, "five", 1L)]
  let duplicate = @device_topology.LocalPeerAccess::new(
    source_ordinal=2,
    destination_ordinal=5,
  )
  try
    @device_topology.LocalTopologyProbeProjection::from_observations(
      driver_version=12080,
      devices,
      [duplicate, duplicate],
    )
  catch {
    InvalidLocalTopology(area=ProbeLinks, rule=Duplicate, ..) => ()
    _ => fail("unexpected duplicate peer rejection")
  } noraise {
    _ => fail("duplicate peer observation was admitted")
  }
  try
    @device_topology.LocalTopologyProbeProjection::from_observations(
      driver_version=12080,
      devices,
      [
        @device_topology.LocalPeerAccess::new(
          source_ordinal=2,
          destination_ordinal=9,
        ),
      ],
    )
  catch {
    InvalidLocalTopology(area=ProbeLinks, rule=Unsupported, ..) => ()
    _ => fail("unexpected ambient peer rejection")
  } noraise {
    _ => fail("ambient peer observation was admitted")
  }
}

///|
test "admission rejects identity target connectivity and count drift" {
  let declaration = two_rank_declaration(topology_target())
  for
    projection in [
      two_rank_projection(second_name="replacement"),
      two_rank_projection(second_minor=6),
    ] {
    try @device_topology.admit_local_topology(declaration, projection) catch {
      InvalidLocalTopology(area=AdmissionRanks, ..) => ()
      _ => fail("unexpected rank drift rejection")
    } noraise {
      _ => fail("rank drift was admitted")
    }
  }
  try
    @device_topology.admit_local_topology(
      declaration,
      two_rank_projection(reverse_peer=false),
    )
  catch {
    InvalidLocalTopology(area=AdmissionLinks, rule=Unsupported, ..) => ()
    _ => fail("unexpected unsupported-connectivity rejection")
  } noraise {
    _ => fail("one-way topology was admitted")
  }
  let extra_projection = @device_topology.LocalTopologyProbeProjection::from_observations(
    driver_version=12080,
    [
      topology_device(2, "rank-two", 17179869184L),
      topology_device(5, "rank-five", 17179869184L),
      topology_device(9, "rank-nine", 17179869184L),
    ],
    [],
  )
  try
    @device_topology.admit_local_topology(declaration, extra_projection)
  catch {
    InvalidLocalTopology(area=AdmissionRanks, rule=CountMismatch, ..) => ()
    _ => fail("unexpected ambient-device rejection")
  } noraise {
    _ => fail("ambient device set was admitted")
  }
}

///|
test "production projection fails closed on unavailable physical inventory" {
  match @device.probe() {
    Unavailable(_) =>
      try
        @device_topology.LocalTopologyProbeProjection::from_local_inventory()
      catch {
        InvalidLocalTopology(area=ProbeDevices, rule=ProbeUnavailable, ..) => ()
        error => fail("unexpected production probe error: \{Repr(error)}")
      } noraise {
        _ => fail("unavailable host published a local topology")
      }
    Available(_) => ()
  }
}

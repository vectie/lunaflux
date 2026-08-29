///|
fn selection_projection(
  second_target? : @catalog.DeviceTarget,
  reverse_peer? : Bool = true,
) -> LocalTopologyProbeProjection raise {
  let target = try! @catalog.DeviceTarget::new(
    compute_major=8,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
  let other = second_target.unwrap_or(target)
  let peers : Array[LocalPeerAccess] = [
    LocalPeerAccess::new(source_ordinal=2, destination_ordinal=5),
  ]
  if reverse_peer {
    peers.push(LocalPeerAccess::new(source_ordinal=5, destination_ordinal=2))
  }
  LocalTopologyProbeProjection::from_observations(
    driver_version=12080,
    [
      LocalProbeDevice::new(
        ordinal=2,
        name="rank-two",
        total_memory_bytes=2147483648L,
        target~,
      ),
      LocalProbeDevice::new(
        ordinal=5,
        name="rank-five",
        total_memory_bytes=2147483648L,
        target=other,
      ),
    ],
    peers,
  )
}

///|
test "rank-ordered topology binds exact ordinals target and full peer mesh" {
  let target = selection_projection().devices[0].target
  let admitted = rank_ordered_topology_from_projection(
    [2, 5],
    target,
    selection_projection(),
  )
  assert_eq(admitted.declaration().rank(0).unwrap().ordinal(), 2)
  assert_eq(admitted.declaration().rank(1).unwrap().ordinal(), 5)
  assert_eq(admitted.topology().world_size(), 2)
  assert_eq(admitted.topology().link_count(), 1)

  let mixed = try! @catalog.DeviceTarget::new(
    compute_major=7,
    compute_minor=5,
    supports_bf16=false,
    supports_cublas_lt=false,
  )
  for
    projection in [
      selection_projection(second_target=mixed),
      selection_projection(reverse_peer=false),
    ] {
    try
      rank_ordered_topology_from_projection([2, 5], target, projection)
    catch {
      InvalidLocalTopology(..) => ()
    } noraise {
      _ => fail("hostile rank-ordered topology was admitted")
    }
  }
}

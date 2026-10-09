///|
fn node(
  name : String,
  minor? : Int = 1,
  ordinal? : Int = 0,
) -> @network_topology.NetworkRank raise {
  @network_topology.NetworkRank::new(
    node=@network_topology.NodeIdentity::new(name),
    device_ordinal=ordinal,
    target=@catalog.DeviceTarget::new(
      compute_major=12,
      compute_minor=minor,
      supports_bf16=true,
      supports_cublas_lt=true,
    ),
    physical_memory_bytes=128000000000L,
    rdma_interface="mlx5_0",
  )
}

///|
test "node-qualified ordinal zero is valid on both nodes and order is retained" {
  let a = node("spark-a")
  let b = node("spark-b")
  let topology = @network_topology.TwoNodeTopology::new(a, b)
  assert_eq(topology.rank(0), Some(a))
  assert_eq(topology.rank(1), Some(b))
  assert_eq(topology.rank(-1), None)
  assert_eq(topology.rank(2), None)
  assert_eq(topology.rank(0).unwrap().device_ordinal(), 0)
  assert_eq(topology.rank(1).unwrap().device_ordinal(), 0)
  assert_eq(topology.target().compute_minor(), 1)
  assert_eq(@network_topology.TwoNodeTopology::new(b, a).rank(0), Some(b))
}

///|
test "two devices on one node cannot masquerade as two network nodes" {
  try
    @network_topology.TwoNodeTopology::new(
      node("same"),
      node("same", ordinal=1),
    )
  catch {
    @network_topology.Invalid(DuplicateNode) => ()
    error => fail("unexpected: \{Repr(error)}")
  } noraise {
    _ => fail("same node accepted")
  }
}

///|
test "adjacent GPU targets are not interchangeable" {
  try
    @network_topology.TwoNodeTopology::new(node("a"), node("b", minor=0))
  catch {
    @network_topology.Invalid(TargetMismatch) => ()
    error => fail("unexpected: \{Repr(error)}")
  } noraise {
    _ => fail("mixed targets accepted")
  }
}

///|
test "node labels reject unbounded and endpoint-like input" {
  for value in ["", "a b", "a/b", "host:22", String::make(129, 'a')] {
    try @network_topology.NodeIdentity::new(value) catch {
      Invalid(NodeIdentity) => ()
      error => fail("unexpected: \{Repr(error)}")
    } noraise {
      _ => fail("invalid label accepted")
    }
  }
  assert_eq(
    @network_topology.NodeIdentity::new("spark-a.1").label(),
    "spark-a.1",
  )
}

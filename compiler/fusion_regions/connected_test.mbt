///|
fn connected_fixture() -> (
  @fusion_regions.Graph,
  Array[@fusion_regions.RegionImplementation],
) raise {
  let graph = @fusion_regions.Graph::new([
    @fusion_regions.Node::new([], observable=false, effect=Ordered),
    @fusion_regions.Node::new([0], observable=true, effect=Ordered),
  ])
  let regions = [
    @fusion_regions.RegionImplementation::new(
      id=10,
      @fusion_regions.Region::new(graph, start=0, count=1, [0]),
      input=Materialized,
      output=Scoped(1),
      workspace_bytes=8L,
      local_bytes=Some(16L),
    ),
    @fusion_regions.RegionImplementation::new(
      id=20,
      @fusion_regions.Region::new(graph, start=1, count=1, [1]),
      input=Scoped(1),
      output=Materialized,
      workspace_bytes=12L,
      local_bytes=Some(8L),
    ),
  ]
  (graph, regions)
}

///|
test "connected choice retains both executable identities and uses whole-chain cost" {
  let (graph, regions) = connected_fixture()
  let alternate = @fusion_regions.RegionImplementation::new(
    id=11,
    @fusion_regions.Region::new(graph, start=0, count=1, [0]),
    input=Materialized,
    output=Scoped(1),
    workspace_bytes=0L,
    local_bytes=Some(8L),
  )
  let plans = [
    @fusion_regions.ConnectedPlan::new(id=1, graph, regions),
    @fusion_regions.ConnectedPlan::new(id=2, graph, [alternate, regions[1]]),
  ]
  // A locally slower producer may win the complete chain. There is no local
  // pruning input here; timings belong to the entire executable cover.
  let costs = [
    @fusion_regions.Cost::measured(nanoseconds=20L, samples=3),
    @fusion_regions.Cost::measured(nanoseconds=10L, samples=3),
  ]
  regions.clear()
  for order in [[0, 1], [1, 0]] {
    let winner = @fusion_regions.select_connected_plan(
      order.map(i => plans[i]),
      order.map(i => costs[i]),
      maximum_workspace_bytes=12L,
      maximum_local_bytes=Some(16L),
    )
    assert_eq(winner.implementation_ids(), [11, 20])
  }
  try
    @fusion_regions.select_connected_plan(
      plans,
      costs,
      maximum_workspace_bytes=11L,
      maximum_local_bytes=Some(16L),
    )
  catch {
    @fusion_regions.NoFeasiblePartition => ()
    e => raise e
  } noraise {
    _ => fail("ignored consumer scratch")
  }
}

///|
test "connected plan rejects missing conversion, non-cover and mixed measurement basis" {
  let (graph, regions) = connected_fixture()
  let incompatible = @fusion_regions.RegionImplementation::new(
    id=30,
    @fusion_regions.Region::new(graph, start=1, count=1, [1]),
    input=Scoped(2),
    output=Materialized,
    workspace_bytes=0L,
    local_bytes=Some(0L),
  )
  for
    chain in [
      [],
      [regions[0]],
      [regions[1], regions[0]],
      [regions[0], incompatible],
    ] {
    try @fusion_regions.ConnectedPlan::new(id=0, graph, chain) catch {
      _ => ()
    } noraise {
      _ => fail("invalid connection accepted")
    }
  }
  let plan = @fusion_regions.ConnectedPlan::new(id=0, graph, regions)
  for costs in [[], [@fusion_regions.Cost::estimated(nanoseconds=1L)]] {
    try
      @fusion_regions.select_connected_plan(
        [plan],
        costs,
        maximum_workspace_bytes=100L,
        maximum_local_bytes=Some(100L),
      )
    catch {
      @fusion_regions.InvalidComparison => ()
      e => raise e
    } noraise {
      _ => fail("unmeasured connection accepted")
    }
  }
}

///|
test "unknown consumer residency is never certified as zero local memory" {
  let (graph, regions) = connected_fixture()
  let external = @fusion_regions.RegionImplementation::new(
    id=40,
    @fusion_regions.Region::new(graph, start=1, count=1, [1]),
    input=Scoped(1),
    output=Materialized,
    workspace_bytes=12L,
    local_bytes=None,
  )
  let plan = @fusion_regions.ConnectedPlan::new(id=0, graph, [
    regions[0],
    external,
  ])
  let cost = @fusion_regions.Cost::measured(nanoseconds=10L, samples=3)
  try
    @fusion_regions.select_connected_plan(
      [plan],
      [cost],
      maximum_workspace_bytes=12L,
      maximum_local_bytes=Some(1000000L),
    )
  catch {
    @fusion_regions.NoFeasiblePartition => ()
    e => raise e
  } noraise {
    _ => fail("opaque resource certified")
  }
  let selected = @fusion_regions.select_connected_plan(
    [plan],
    [cost],
    maximum_workspace_bytes=12L,
    maximum_local_bytes=None,
  )
  assert_eq(selected.implementation_ids(), [10, 40])
}

///|
fn chain() -> @fusion_regions.Graph raise {
  @fusion_regions.Graph::new([
    @fusion_regions.Node::new([], observable=false, effect=Pure),
    @fusion_regions.Node::new([0], observable=false, effect=Pure),
    @fusion_regions.Node::new([1], observable=false, effect=Ordered),
    @fusion_regions.Node::new([2], observable=true, effect=Ordered),
  ])
}

///|
fn candidate(
  graph : @fusion_regions.Graph,
  id : Int,
  start : Int,
  count : Int,
  ns : Int64,
  local_bytes? : Int64 = 0L,
) -> @fusion_regions.Candidate raise {
  @fusion_regions.Candidate::new(
    id~,
    @fusion_regions.Region::new(graph, start~, count~, [start + count - 1]),
    @fusion_regions.Cost::estimated(nanoseconds=ns),
    workspace_bytes=0L,
    local_bytes~,
  )
}

///|
test "arbitrary DAG fanout and observable values survive every region boundary" {
  let graph = @fusion_regions.Graph::new([
    @fusion_regions.Node::new([], observable=false, effect=Pure),
    @fusion_regions.Node::new([0], observable=true, effect=Pure),
    @fusion_regions.Node::new([0], observable=false, effect=Pure),
    @fusion_regions.Node::new([1, 2], observable=false, effect=Ordered),
    @fusion_regions.Node::new([0, 3], observable=false, effect=Boundary),
    @fusion_regions.Node::new([3, 4], observable=true, effect=Ordered),
  ])
  for exports in [[3], [0, 3], [1, 3]] {
    try @fusion_regions.Region::new(graph, start=0, count=4, exports) catch {
      @fusion_regions.MissingExport => ()
      _ => fail("wrong missing output rejection")
    } noraise {
      _ => fail("shared or observable value was hidden")
    }
  }
  let first = @fusion_regions.Region::new(graph, start=0, count=4, [0, 1, 3])
  assert_eq(first.exports(), [0, 1, 3])
  assert_eq(first.effect_order(), [3])
  try @fusion_regions.Region::new(graph, start=4, count=2, [5]) catch {
    @fusion_regions.EffectBoundary => ()
    _ => fail("wrong effect error")
  } noraise {
    _ => fail("externally ordered boundary was fused")
  }
  let partition = @fusion_regions.Partition::new(graph, [
    first,
    @fusion_regions.Region::new(graph, start=4, count=1, [4]),
    @fusion_regions.Region::new(graph, start=5, count=1, [5]),
  ])
  assert_eq(partition.regions().length(), 3)
}

///|
test "global region choice beats greedy largest fusion and preserves effects" {
  let graph = chain()
  let candidates = [
    candidate(graph, 0, 0, 4, 10L),
    candidate(graph, 1, 0, 1, 2L),
    candidate(graph, 2, 1, 3, 2L),
    candidate(graph, 3, 0, 2, 1L),
    candidate(graph, 4, 2, 2, 20L),
  ]
  for order in [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [2, 4, 0, 3, 1]] {
    let selected = @fusion_regions.select_regions(
      graph,
      order.map(index => candidates[index]),
      maximum_workspace_bytes=0L,
      maximum_local_bytes=0L,
    )
    assert_eq(selected.implementation_ids(), [1, 2])
    assert_eq(selected.cost().nanoseconds(), 4L)
    assert_eq(selected.cost().basis(), @fusion_regions.Estimated)
    assert_eq(selected.partition().regions()[1].effect_order(), [2, 3])
  }
}

///|
test "stable catalog identity breaks full-path cost ties independent of enumeration" {
  let graph = chain()
  let left = candidate(graph, 2, 0, 2, 2L)
  let right = candidate(graph, 3, 2, 2, 2L)
  let whole = candidate(graph, 4, 0, 4, 4L)
  for candidates in [[whole, right, left], [left, whole, right]] {
    assert_eq(
      @fusion_regions.select_regions(
        graph,
        candidates,
        maximum_workspace_bytes=0L,
        maximum_local_bytes=0L,
      ).implementation_ids(),
      [2, 3],
    )
  }
}

///|
test "resource filtering applies before cost ranking with no implicit slow fallback" {
  let graph = chain()
  let whole = candidate(graph, 0, 0, 4, 1L, local_bytes=64L)
  let permitted = candidate(graph, 1, 0, 4, 10L, local_bytes=32L)
  assert_eq(
    @fusion_regions.select_regions(
      graph,
      [whole, permitted],
      maximum_workspace_bytes=0L,
      maximum_local_bytes=32L,
    ).implementation_ids(),
    [1],
  )
  try
    @fusion_regions.select_regions(
      graph,
      [whole],
      maximum_workspace_bytes=0L,
      maximum_local_bytes=32L,
    )
  catch {
    @fusion_regions.NoFeasiblePartition => ()
    _ => fail("wrong resource error")
  } noraise {
    _ => fail("oversized implementation accepted")
  }
}

///|
test "malformed graph exports partitions and mismatched comparison bases are rejected" {
  let repeated_operand = @fusion_regions.Graph::new([
    @fusion_regions.Node::new([], observable=false, effect=Pure),
    @fusion_regions.Node::new([0, 0], observable=true, effect=Pure),
  ])
  assert_eq(repeated_operand.node_count(), 2)
  for inputs in [[0], [-1], [1]] {
    try
      @fusion_regions.Graph::new([
        @fusion_regions.Node::new(inputs, observable=true, effect=Pure),
      ])
    catch {
      @fusion_regions.InvalidGraph => ()
      _ => fail("wrong graph error")
    } noraise {
      _ => fail("cycle or invalid SSA edge accepted")
    }
  }
  let graph = chain()
  for exports in [[3, 3], [4], [-1], [3, 0]] {
    try @fusion_regions.Region::new(graph, start=0, count=4, exports) catch {
      @fusion_regions.InvalidRegion => ()
      _ => fail("wrong export error")
    } noraise {
      _ => fail("malformed exports accepted")
    }
  }
  let first = @fusion_regions.Region::new(graph, start=0, count=2, [1])
  try @fusion_regions.Partition::new(graph, [first, first]) catch {
    @fusion_regions.InvalidPartition => ()
    _ => fail("wrong partition error")
  } noraise {
    _ => fail("overlapping or missing effects accepted")
  }
  let region = @fusion_regions.Region::new(graph, start=0, count=4, [3])
  let measured = @fusion_regions.Candidate::new(
    id=1,
    region,
    @fusion_regions.Cost::measured(nanoseconds=1L, samples=3),
    workspace_bytes=0L,
    local_bytes=0L,
  )
  let estimate = candidate(graph, 2, 0, 4, 1L)
  for candidates in [[measured, estimate], [measured, measured]] {
    try
      @fusion_regions.select_regions(
        graph,
        candidates,
        maximum_workspace_bytes=0L,
        maximum_local_bytes=0L,
      )
    catch {
      @fusion_regions.InvalidComparison => ()
      _ => fail("wrong comparison error")
    } noraise {
      _ => fail("invalid comparison accepted")
    }
  }
}

///|
test "whole-chain measurements are not conflated with region timing sums" {
  let graph = chain()
  let whole = @fusion_regions.Partition::new(graph, [
    @fusion_regions.Region::new(graph, start=0, count=4, [3]),
  ])
  let split = @fusion_regions.Partition::new(graph, [
    @fusion_regions.Region::new(graph, start=0, count=2, [1]),
    @fusion_regions.Region::new(graph, start=2, count=2, [3]),
  ])
  let choices = [
    @fusion_regions.PartitionAlternative::new(
      id=2,
      whole,
      @fusion_regions.Cost::measured(nanoseconds=10L, samples=3),
      workspace_bytes=64L,
      local_bytes=0L,
    ),
    @fusion_regions.PartitionAlternative::new(
      id=1,
      split,
      @fusion_regions.Cost::measured(nanoseconds=10L, samples=3),
      workspace_bytes=0L,
      local_bytes=0L,
    ),
  ]
  assert_eq(
    @fusion_regions.select_partition(
      choices,
      maximum_workspace_bytes=64L,
      maximum_local_bytes=0L,
    ),
    1,
  )
  assert_eq(
    @fusion_regions.select_partition(
      choices,
      maximum_workspace_bytes=0L,
      maximum_local_bytes=0L,
    ),
    1,
  )
}

///|
test "zero under-sampled and overflowing composed costs cannot win" {
  try @fusion_regions.Cost::measured(nanoseconds=1L, samples=2) catch {
    @fusion_regions.InvalidCost => ()
    _ => fail("wrong cost error")
  } noraise {
    _ => fail("under-sampled cost")
  }
  try @fusion_regions.Cost::estimated(nanoseconds=0L) catch {
    @fusion_regions.InvalidCost => ()
    _ => fail("wrong cost error")
  } noraise {
    _ => fail("zero cost")
  }
  let graph = chain()
  try
    @fusion_regions.select_regions(
      graph,
      [
        candidate(graph, 1, 0, 2, 9223372036854775807L),
        candidate(graph, 2, 2, 2, 1L),
      ],
      maximum_workspace_bytes=0L,
      maximum_local_bytes=0L,
    )
  catch {
    @fusion_regions.InvalidCost => ()
    _ => fail("wrong overflow error")
  } noraise {
    _ => fail("cost sum overflow")
  }
}

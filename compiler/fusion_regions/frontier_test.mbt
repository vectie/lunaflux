///|
fn complete_partition() -> @fusion_regions.Partition raise {
  let graph = @fusion_regions.Graph::new([
    @fusion_regions.Node::new([], observable=false, effect=Pure),
    @fusion_regions.Node::new([0], observable=true, effect=Ordered),
  ])
  @fusion_regions.Partition::new(graph, [
    @fusion_regions.Region::new(graph, start=0, count=2, [1]),
  ])
}

///|
fn alternative(
  partition : @fusion_regions.Partition,
  id : Int,
  ns : Int64,
  workspace : Int64,
  local_storage : Int64,
  continuation? : @fusion_regions.ContinuationClass = Materialized,
) -> @fusion_regions.PartitionAlternative raise {
  @fusion_regions.PartitionAlternative::new(
    id~,
    partition,
    @fusion_regions.Cost::estimated(nanoseconds=ns),
    workspace_bytes=workspace,
    local_bytes=local_storage,
    continuation_class=continuation,
  )
}

///|
fn retained_ids(
  frontier : @fusion_regions.PartitionFrontier,
) -> ReadOnlyArray[Int] {
  frontier.alternatives().map(value => value.implementation_id())
}

///|
test "retain all resource tradeoffs and resolve the final envelope without mutation" {
  let partition = complete_partition()
  let choices = [
    alternative(partition, 1, 10L, 64L, 32L),
    alternative(partition, 2, 20L, 0L, 32L),
    alternative(partition, 3, 30L, 64L, 0L),
    alternative(partition, 4, 40L, 0L, 0L),
    alternative(partition, 5, 50L, 64L, 32L),
  ]
  let frontier = @fusion_regions.PartitionFrontier::new(choices)
  assert_eq(retained_ids(frontier), [1, 2, 3, 4])
  // Caller storage is copied; changing its enumeration cannot change the plan.
  choices.clear()
  for
    (workspace, local_storage, expected) in [
      (64L, 32L, 1),
      (0L, 32L, 2),
      (64L, 0L, 3),
      (0L, 0L, 4),
    ] {
    let selected = frontier.select(
      continuation_class=Materialized,
      maximum_workspace_bytes=workspace,
      maximum_local_bytes=local_storage,
    )
    assert_eq(selected.implementation_id(), expected)
    assert_eq(selected.partition().regions()[0].effect_order(), [1])
    assert_eq(selected.partition().regions()[0].exports(), [1])
  }
  assert_eq(retained_ids(frontier), [1, 2, 3, 4])
}

///|
test "a slower producer with another continuation contract cannot be pruned" {
  let partition = complete_partition()
  let choices = [
    alternative(partition, 1, 1L, 0L, 0L),
    alternative(partition, 2, 20L, 32L, 16L, continuation=Scoped(7)),
    alternative(partition, 3, 30L, 32L, 16L, continuation=Scoped(7)),
    alternative(partition, 4, 1L, 0L, 0L, continuation=Scoped(0)),
  ]
  let frontier = @fusion_regions.PartitionFrontier::new(choices)
  assert_eq(retained_ids(frontier), [1, 2, 4])
  assert_eq(
    frontier
    .select(
      continuation_class=Scoped(0),
      maximum_workspace_bytes=0L,
      maximum_local_bytes=0L,
    )
    .implementation_id(),
    4,
  )
  let selected = frontier.select(
    continuation_class=Scoped(7),
    maximum_workspace_bytes=32L,
    maximum_local_bytes=16L,
  )
  assert_eq(selected.implementation_id(), 2)
  assert_eq(selected.continuation_class(), @fusion_regions.Scoped(7))
  assert_eq(selected.cost().nanoseconds(), 20L)
  assert_eq(selected.resources(), (32L, 16L))
  assert_eq(
    @fusion_regions.select_partition(
      choices,
      continuation_class=Scoped(7),
      maximum_workspace_bytes=32L,
      maximum_local_bytes=16L,
    ),
    2,
  )
  for requested in [7, 8] {
    try
      frontier.select(
        continuation_class=Scoped(requested),
        maximum_workspace_bytes=0L,
        maximum_local_bytes=0L,
      )
    catch {
      @fusion_regions.NoFeasiblePartition => ()
      _ => fail("wrong unavailable continuation error")
    } noraise {
      _ => fail("silently fell back to a different continuation")
    }
  }
}

///|
test "pruning preserves stable latency ties and is permutation invariant" {
  let partition = complete_partition()
  let choices = [
    alternative(partition, 1, 10L, 32L, 0L),
    alternative(partition, 2, 10L, 0L, 0L),
    alternative(partition, 3, 10L, 0L, 0L),
  ]
  for order in [[0, 1, 2], [2, 1, 0], [1, 0, 2]] {
    let frontier = @fusion_regions.PartitionFrontier::new(
      order.map(i => choices[i]),
    )
    assert_eq(retained_ids(frontier), [1, 2])
    for (workspace, expected) in [(32L, 1), (0L, 2)] {
      assert_eq(
        frontier
        .select(
          continuation_class=Materialized,
          maximum_workspace_bytes=workspace,
          maximum_local_bytes=0L,
        )
        .implementation_id(),
        expected,
      )
    }
  }
}

///|
test "frontier agrees with exhaustive selection over budgets and continuation classes" {
  let partition = complete_partition()
  let choices = [
    for id in 0..<60 => {
      alternative(
        partition,
        id,
        (id * 7 % 11 + 1).to_int64(),
        (id % 4).to_int64(),
        (id % 5).to_int64(),
        continuation=Scoped(id % 3),
      )
    }
  ]
  let frontier = @fusion_regions.PartitionFrontier::new(choices)
  for continuation in 0..<3 {
    for workspace in 0..<4 {
      for local_storage in 0..<5 {
        // Independent exhaustive oracle: no dominance or frontier logic.
        let expected = for value in choices; best = (None : (Int64, Int)?) {
          let (scratch, storage) = value.resources()
          if value.continuation_class() != @fusion_regions.Scoped(continuation) ||
            scratch > workspace.to_int64() ||
            storage > local_storage.to_int64() {
            continue best
          }
          let ns = value.cost().nanoseconds()
          let id = value.implementation_id()
          continue match best {
              Some((old_ns, old_id)) if old_ns < ns ||
                (old_ns == ns && old_id < id) => best
              _ => Some((ns, id))
            }
        } nobreak {
          best
        }
        let actual = try
          frontier.select(
            continuation_class=Scoped(continuation),
            maximum_workspace_bytes=workspace.to_int64(),
            maximum_local_bytes=local_storage.to_int64(),
          )
        catch {
          @fusion_regions.NoFeasiblePartition => None
          other => raise other
        } noraise {
          value => Some((value.cost().nanoseconds(), value.implementation_id()))
        }
        assert_eq(actual, expected)
      }
    }
  }
}

///|
test "frontier rejects mixed graphs bases duplicate identities and invalid budgets" {
  let partition = complete_partition()
  let valid = alternative(partition, 1, 10L, 0L, 0L)
  let measured = @fusion_regions.PartitionAlternative::new(
    id=2,
    partition,
    @fusion_regions.Cost::measured(nanoseconds=1L, samples=3),
    workspace_bytes=0L,
    local_bytes=0L,
  )
  let foreign_graph = @fusion_regions.Graph::new([
    @fusion_regions.Node::new([], observable=true, effect=Boundary),
  ])
  let foreign = @fusion_regions.Partition::new(foreign_graph, [
    @fusion_regions.Region::new(foreign_graph, start=0, count=1, [0]),
  ])
  for
    choices in [
      [],
      [valid, valid],
      [valid, measured],
      [valid, alternative(foreign, 3, 1L, 0L, 0L)],
    ] {
    try @fusion_regions.PartitionFrontier::new(choices) catch {
      @fusion_regions.InvalidComparison => ()
      _ => fail("wrong comparison rejection")
    } noraise {
      _ => fail("invalid comparison accepted")
    }
  }
  let frontier = @fusion_regions.PartitionFrontier::new([valid])
  for
    (continuation, workspace, local_storage) in [
      (-1, 0L, 0L),
      (0, -1L, 0L),
      (0, 0L, -1L),
    ] {
    try
      frontier.select(
        continuation_class=Scoped(continuation),
        maximum_workspace_bytes=workspace,
        maximum_local_bytes=local_storage,
      )
    catch {
      @fusion_regions.InvalidComparison => ()
      _ => fail("wrong budget rejection")
    } noraise {
      _ => fail("invalid budget accepted")
    }
  }
  try alternative(partition, 1, 10L, 0L, 0L, continuation=Scoped(-1)) catch {
    @fusion_regions.InvalidCost => ()
    _ => fail("wrong class rejection")
  } noraise {
    _ => fail("negative continuation class")
  }
}

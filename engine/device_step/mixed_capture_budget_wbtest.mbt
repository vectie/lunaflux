///|
test "mandatory mixed captures reserve once per physical domain before optional graphs" {
  let (table, _, _) = graph_dispatch_fixture(0)
  let ids = Array::make(table.bucket_count(), -1)
  let a = graph_dispatch_bucket(table, MixedGraph, 2, 64, 4096)
  let b = graph_dispatch_bucket(table, MixedGraph, 2, 64, 8192)
  ids[a.slot()] = 7
  ids[b.slot()] = 7
  let budget = @luna_execution_graph_strategy.ExecutionGraphVariantBudget::new(
    2, 10,
  )
  let remaining = reserve_measured_mixed_capture_budget(
    budget,
    table,
    Some(ReadOnlyArray::from_array(ids)),
    5,
  )
  // Different histories share one executable domain, leaving exactly one
  // optional graph. Consuming it cannot invalidate the mandatory reservation.
  assert_eq(
    remaining.reserve(5),
    Some(
      try! @luna_execution_graph_strategy.ExecutionGraphVariantBudget::new(0, 0),
    ),
  )
  assert_eq(remaining.reserve(6), None)
  assert_eq(remaining.reserve(5).unwrap().reserve(1), None)
  let c = graph_dispatch_bucket(table, MixedGraph, 16, 64, 4096)
  ids[c.slot()] = 7
  assert_eq(
    reserve_measured_mixed_capture_budget(
      budget,
      table,
      Some(ReadOnlyArray::from_array(ids)),
      5,
    ),
    try! @luna_execution_graph_strategy.ExecutionGraphVariantBudget::new(0, 0),
  )
}

///|
test "mixed capture reservation rejects missing capacity and wrong phases without widening budgets" {
  let (table, _, _) = graph_dispatch_fixture(0)
  let ids = Array::make(table.bucket_count(), -1)
  let bucket = graph_dispatch_bucket(table, MixedGraph, 2, 64, 4096)
  ids[bucket.slot()] = 7
  let candidates = Some(ReadOnlyArray::from_array(ids))
  for
    budget in [
      effect_capture_budget(GraphMemoryBounded),
      effect_capture_budget(GraphMemoryUnsupported),
    ] {
    let rejected = reserve_measured_mixed_capture_budget(
      budget, table, candidates, 5,
    ) catch {
      InvalidExecutor(GraphBucket) => {
        assert_eq(budget.reserve(1), None)
        continue
      }
      _ => fail("unexpected reservation failure")
    }
    ignore(rejected)
    fail("zero-capacity measured capture admitted")
  }
  ids[bucket.slot()] = -1
  ids[graph_dispatch_bucket(table, PrefillGraph, 2, 64, 4096).slot()] = 7
  let rejected = reserve_measured_mixed_capture_budget(
    effect_capture_budget(GraphMemoryAbsent),
    table,
    Some(ReadOnlyArray::from_array(ids)),
    5,
  ) catch {
    InvalidExecutor(GraphBucket) => return
    _ => fail("unexpected phase failure")
  }
  ignore(rejected)
  fail("prefill relabeled as mixed")
}

///|
test "column fold windows cover independent outputs exactly once under a live budget" {
  for tiles in [1, 2, 3, 4, 6, 8, 9, 31] {
    for groups in [1, 2, 4, 8] {
      for budget in [1, 2, 3, 8] {
        let plan = ProjectionColumnFoldPlan::new(
          output_tiles=tiles,
          consumer_groups=groups,
          maximum_live_accumulators=budget,
        ).unwrap()
        let (owners, rounds, live, windows) = plan.geometry()
        assert_eq(owners, groups)
        assert_eq(rounds, (tiles - 1) / groups + 1)
        assert_true(live > 0 && live <= budget)
        let visited = Array::make(tiles, 0)
        for window in 0..<windows {
          for slot in 0..<live {
            for owner in 0..<owners {
              if plan.column_at(window, slot, owner) is Some(column) {
                assert_eq(column, (window * live + slot) * groups + owner)
                visited[column] += 1
              }
            }
          }
        }
        for visits in visited {
          assert_eq(visits, 1)
        }
        assert_eq(plan.column_at(-1, 0, 0), None)
        assert_eq(plan.column_at(windows, 0, 0), None)
        assert_eq(plan.column_at(0, live, 0), None)
        assert_eq(plan.column_at(0, 0, owners), None)
      }
    }
  }
}

///|
test "column fold plans reject invalid dimensions and avoid tail arithmetic overflow" {
  for (tiles, groups, budget) in [(0, 4, 2), (4, 0, 2), (4, 4, 0), (-1, 4, 2)] {
    assert_eq(
      ProjectionColumnFoldPlan::new(
        output_tiles=tiles,
        consumer_groups=groups,
        maximum_live_accumulators=budget,
      ),
      None,
    )
  }
  let plan = ProjectionColumnFoldPlan::new(
    output_tiles=2147483647,
    consumer_groups=4,
    maximum_live_accumulators=3,
  ).unwrap()
  let (_, _, _, windows) = plan.geometry()
  assert_eq(plan.column_at(windows - 1, 1, 2), Some(2147483646))
  assert_eq(plan.column_at(windows - 1, 1, 3), None)
  assert_eq(plan.column_at(windows - 1, 2, 0), None)
}

///|
test "column fold resource envelope changes physical identity without changing tile coverage" {
  let one = ProjectionColumnFoldPlan::new(
    output_tiles=8,
    consumer_groups=4,
    maximum_live_accumulators=1,
  ).unwrap()
  let two = ProjectionColumnFoldPlan::new(
    output_tiles=8,
    consumer_groups=4,
    maximum_live_accumulators=2,
  ).unwrap()
  assert_eq(one.geometry(), (4, 2, 1, 2))
  assert_eq(two.geometry(), (4, 2, 2, 1))
  assert_false(one.canonical_bytes() == two.canonical_bytes())
  assert_true(
    @utf8.decode(two.canonical_bytes()).contains("reduction=ordered\n"),
  )
}

///|
test "scheduled dense projection refines to a generic column fold plan" {
  let compilation = compile_projection_tile(
    ProjectionTileCompilerRequest::new(
      dispatch=projection_compiler_dispatch(Dense),
      maximum_rows=32,
      maximum_tokens=256,
      demand=AllTokenRows,
    ),
  )
  assert_eq(
    compilation
    .schedule()
    .column_fold_plan(consumer_groups=4, maximum_live_accumulators=2)
    .unwrap()
    .geometry(),
    (4, 16, 2, 8),
  )
}

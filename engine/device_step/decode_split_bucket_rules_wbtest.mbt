///|
test "split attention preserves following operation bucket rules after expansion" {
  let launch = variant_test_launch(45056)
  let make = fn(
    operation : Int,
    function_index : Int,
  ) -> (@plan.OperationId, Int) raise {
    (@plan.OperationId::new(operation), function_index)
  }
  let original = ReadOnlyArray::from_array([make(1, 1), make(2, 2), make(3, 3)])
  let expanded = ReadOnlyArray::from_array([
    make(1, 1),
    make(2, 20),
    make(2, 21),
    make(3, 3),
  ])
  let contract : PagedGraphLaunchContract = {
    step_launches: [FixedBucketLaunch, FixedBucketLaunch, FixedBucketLaunch],
    decode_step_launches: [
      QueryTokenCappedGridX(launch),
      QueryTileCappedGridX(launch, 32),
      QueryTokenCappedGridX(launch),
    ],
    prefill_wide_step_launches: None,
    prefill_split_step_launches: None,
  }
  let rules = remap_decode_split_bucket_rules(original, expanded, contract)
  assert_eq(rules.length(), 4)
  assert_true(rules[1] is FixedBucketLaunch)
  assert_true(rules[2] is FixedBucketLaunch)
  for index in [0, 3] {
    assert_true(rules[index] is QueryTokenCappedGridX(_))
    for bound in [1, 8, 16, 32] {
      assert_eq(paged_bucket_launch(rules[index], bound).grid_x(), bound)
      assert_eq(
        paged_bucket_launch(rules[index], bound).shared_memory_bytes(),
        45056,
      )
    }
  }
}

///|
test "admitted split row domains reach both partial and merge capture buckets" {
  let operation = @plan.OperationId::new(2)
  let launch = variant_test_launch(45056)
  let merge = variant_test_launch(0)
  let original = ReadOnlyArray::from_array([(operation, 2)])
  let expanded = ReadOnlyArray::from_array([(operation, 20), (operation, 21)])
  let contract : PagedGraphLaunchContract = {
    step_launches: [FixedBucketLaunch],
    decode_step_launches: [QueryTileCappedGridX(launch, 32)],
    prefill_wide_step_launches: None,
    prefill_split_step_launches: None,
  }
  let rules = remap_decode_split_bucket_rules(
    original,
    expanded,
    contract,
    row_bounded_replacements=ReadOnlyArray::from_array([
      ((operation, 20), launch),
      ((operation, 21), merge),
    ]),
  )
  for bound in [1, 8, 16, 32] {
    for index, source in [launch, merge] {
      let selected = paged_bucket_launch(rules[index], bound)
      assert_eq(selected.grid_x(), Int::min(source.grid_x(), bound))
      assert_eq(selected.grid_y(), source.grid_y())
      assert_eq(selected.grid_z(), source.grid_z())
      assert_eq(selected.block_x(), source.block_x())
      assert_eq(selected.shared_memory_bytes(), source.shared_memory_bytes())
    }
  }
  for
    invalid in [
      [((operation, 20), launch), ((operation, 20), merge)],
      [((operation, 2), launch)],
      [((operation, 99), launch)],
    ] {
    try
      remap_decode_split_bucket_rules(
        original,
        expanded,
        contract,
        row_bounded_replacements=ReadOnlyArray::from_array(invalid),
      )
    catch {
      InvalidExecutor(GraphBucket) => ()
      _ => fail("unexpected failure")
    } noraise {
      _ => fail("invalid replacement domain accepted")
    }
  }
}

///|
test "split decode dispatch selects bounded owner and retains safe fallback" {
  let (table, _, base) = graph_dispatch_fixture(7)
  let dispatch : PagedOrderedExecutorDispatch = {
    ..base,
    decode_split_owner_index: 11,
    decode_split_minimum_context_tokens: 128,
    decode_split_maximum_batch_rows: 8,
  }
  assert_eq(
    dispatch.select(graph_dispatch_bucket(table, DecodeGraph, 1, 1, 256)),
    11,
  )
  map_paged_decode_capture_owner(
    table,
    1,
    1,
    21,
    dispatch.decode_split_slot_owner_indices,
  )
  map_paged_decode_capture_owner(
    table,
    8,
    8,
    28,
    dispatch.decode_split_slot_owner_indices,
  )
  assert_eq(
    dispatch.select(graph_dispatch_bucket(table, DecodeGraph, 1, 1, 256)),
    21,
  )
  assert_eq(
    dispatch.select(graph_dispatch_bucket(table, DecodeGraph, 8, 8, 256)),
    28,
  )
  assert_eq(
    dispatch.select(graph_dispatch_bucket(table, DecodeGraph, 1, 1, 64)),
    7,
  )
  assert_eq(
    dispatch.select(graph_dispatch_bucket(table, PrefillGraph, 1, 1, 256)),
    7,
  )
  assert_eq(
    dispatch.select(graph_dispatch_bucket(table, MixedGraph, 1, 1, 256)),
    7,
  )
}

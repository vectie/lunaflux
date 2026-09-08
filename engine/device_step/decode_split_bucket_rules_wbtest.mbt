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

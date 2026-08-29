///|
test "context churn scalar contract is strict and bounded" {
  assert_true(
    context_churn_scalar_contract_valid(
      1031, 2185, 2999, 1, 8, 8, 32, 256, 256, 1024,
    ),
  )
  assert_false(
    context_churn_scalar_contract_valid(
      3000, 2185, 2999, 1, 8, 8, 32, 256, 256, 1024,
    ),
  )
  assert_false(
    context_churn_scalar_contract_valid(
      1031, 2185, 2999, 1, 2, 8, 32, 256, 256, 1024,
    ),
  )
  assert_false(
    context_churn_scalar_contract_valid(
      1031, 2185, 2999, 1, 8, 8, 8, 256, 256, 1024,
    ),
  )
  assert_false(
    context_churn_scalar_contract_valid(
      1031, 2185, 2999, 1, 8, 8, 32, 257, 256, 1024,
    ),
  )
  assert_false(
    context_churn_scalar_contract_valid(
      1031, 2185, 2999, 1, 8, 8, 32, 256, 256, 0,
    ),
  )
}

///|
test "context churn contract-owned frames cannot be caller substituted" {
  let source : FixedArray[Byte] = [b'a', b'b', b'c']
  let owned = copy_context_churn_frame(source)
  source[0] = b'z'
  assert_eq(owned, [b'a', b'b', b'c'])
}

///|
test "context churn telemetry requires exact resource balance" {
  let balanced : SpawnedDirectResources = {
    queue_depth: 0,
    active_requests: 0,
    kv_pages_used: 0,
    kv_pages_free: 32,
    prefix_lookups: 0UL,
    prefix_hits: 0UL,
    prefix_misses: 0UL,
    prefix_publications: 0UL,
    prefix_tokens_reused: 0UL,
    prefix_tokens_computed: 0UL,
    prefix_entries: 0,
    prefix_pages: 0,
  }
  assert_true(spawned_direct_resources_valid(balanced, 32, 0UL))
  let leaked : SpawnedDirectResources = {
    ..balanced,
    kv_pages_used: 1,
    kv_pages_free: 31,
  }
  assert_false(spawned_direct_resources_valid(leaked, 32, 0UL))
  let cached : SpawnedDirectResources = { ..balanced, prefix_lookups: 1UL }
  assert_false(spawned_direct_resources_valid(cached, 32, 0UL))
  let completed : SpawnedDirectResources = {
    ..balanced,
    prefix_tokens_computed: 3UL,
  }
  assert_true(spawned_direct_resources_valid(completed, 32, 3UL))
}

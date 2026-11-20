///|
test "decode capture bounds stop before the maximum envelope" {
  let (_, profile, _) = graph_dispatch_fixture(7)
  let table = new_paged_graph_bucket_table(profile)
  assert_eq(paged_decode_capture_bounds(profile), [1, 8, 16, 32])
  let maximum_slots = paged_maximum_graph_slots(table, profile)
  let live_decode = graph_dispatch_bucket(table, DecodeGraph, 32, 32, 1048576)
  assert_eq(maximum_slots[1], live_decode.slot())
}

///|
test "prefill capture bounds cover powers of two below the maximum envelope" {
  let (_, profile, _) = graph_dispatch_fixture(7)
  assert_eq(paged_prefill_capture_bounds(profile), [1, 8, 16, 32])
  let larger = @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(92),
    max_query_rows=32,
    max_query_tokens=4096,
    max_page_table_entries=256,
  )
  assert_eq(paged_prefill_capture_bounds(larger), [
    1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,
  ])
}

///|
test "decode bucket owners map exact compatible slots to the smallest owner" {
  let (table, _, dispatch) = graph_dispatch_fixture(7)
  map_paged_decode_capture_owner(table, 8, 8, 4, dispatch.slot_owner_indices)
  let one = graph_dispatch_bucket(table, DecodeGraph, 1, 1, 1)
  let eight = graph_dispatch_bucket(table, DecodeGraph, 8, 8, 65536)
  let larger = graph_dispatch_bucket(table, DecodeGraph, 16, 16, 65536)
  assert_eq(dispatch.select(one), 4)
  assert_eq(dispatch.select(eight), 4)
  assert_eq(dispatch.select(larger), 7)
}

///|
test "prefill bucket owners map live token capacities without touching larger buckets" {
  let (table, profile, dispatch) = graph_dispatch_fixture(7)
  map_paged_prefill_capture_owner(
    table,
    profile.max_query_rows(),
    16,
    5,
    dispatch.slot_owner_indices,
  )
  let small = graph_dispatch_bucket(table, PrefillGraph, 4, 12, 256)
  let exact = graph_dispatch_bucket(table, PrefillGraph, 16, 16, 65536)
  let larger = graph_dispatch_bucket(table, PrefillGraph, 16, 32, 65536)
  let decode = graph_dispatch_bucket(table, DecodeGraph, 8, 8, 65536)
  let mixed = graph_dispatch_bucket(table, MixedGraph, 8, 12, 65536)
  let impossible_single_row_mixed = graph_dispatch_bucket(
    table,
    MixedGraph,
    1,
    1,
    65536,
  )
  assert_eq(dispatch.select(small), 5)
  assert_eq(dispatch.select(exact), 5)
  assert_eq(dispatch.select(mixed), 5)
  assert_eq(dispatch.select(impossible_single_row_mixed), 7)
  assert_eq(dispatch.select(larger), 7)
  assert_eq(dispatch.select(decode), 7)
}

///|
test "partitioned prefill scales context threshold with query parallelism" {
  let (table, _, base) = graph_dispatch_fixture(7)
  let dispatch : PagedOrderedExecutorDispatch = {
    ..base,
    prefill_split_owner_index: 13,
    prefill_split_minimum_context_tokens: 512,
    prefill_split_query_tile: 32,
    prefill_split_partition_count: 2,
  }
  assert_eq(
    dispatch.select(graph_dispatch_bucket(table, PrefillGraph, 1, 32, 256)),
    7,
  )
  assert_eq(
    dispatch.select(graph_dispatch_bucket(table, PrefillGraph, 1, 32, 4096)),
    13,
  )
  assert_eq(
    dispatch.select(graph_dispatch_bucket(table, PrefillGraph, 2, 32, 4096)),
    13,
  )
  assert_eq(
    dispatch.select(graph_dispatch_bucket(table, MixedGraph, 2, 32, 4096)),
    13,
  )
  assert_eq(
    dispatch.select(graph_dispatch_bucket(table, DecodeGraph, 8, 8, 4096)),
    7,
  )
  assert_eq(
    dispatch.select(graph_dispatch_bucket(table, PrefillGraph, 4, 64, 1024)),
    7,
  )
  assert_eq(
    dispatch.select(graph_dispatch_bucket(table, PrefillGraph, 4, 64, 4096)),
    13,
  )
  assert_eq(
    dispatch.select(graph_dispatch_bucket(table, PrefillGraph, 8, 64, 4096)),
    7,
  )
}

///|
test "deep partition frontier supersedes base only for enough bucket work" {
  let (table, _, base) = graph_dispatch_fixture(7)
  let dispatch : PagedOrderedExecutorDispatch = {
    ..base,
    prefill_split_owner_index: 13,
    prefill_split_minimum_context_tokens: 256,
    prefill_split_query_tile: 16,
    prefill_split_partition_count: 2,
    deep_prefill_split_owner_index: 17,
    deep_prefill_split_minimum_context_tokens: 1024,
    deep_prefill_split_query_tile: 16,
    deep_prefill_split_partition_count: 8,
  }
  assert_eq(
    dispatch.select(graph_dispatch_bucket(table, PrefillGraph, 1, 16, 1024)),
    13,
  )
  assert_eq(
    dispatch.select(graph_dispatch_bucket(table, PrefillGraph, 1, 16, 32768)),
    13,
  )
  assert_eq(
    dispatch.select(graph_dispatch_bucket(table, PrefillGraph, 1, 16, 65536)),
    17,
  )
  assert_eq(
    dispatch.select(graph_dispatch_bucket(table, MixedGraph, 1, 16, 65536)),
    17,
  )
  assert_eq(
    dispatch.select(graph_dispatch_bucket(table, DecodeGraph, 1, 1, 4096)),
    7,
  )
}

///|
test "partitioned prefill selects the smallest compiled query bucket" {
  let (table, profile, base) = graph_dispatch_fixture(7)
  let dispatch : PagedOrderedExecutorDispatch = {
    ..base,
    prefill_split_owner_index: 15,
    prefill_split_minimum_context_tokens: 512,
    prefill_split_query_tile: 32,
    prefill_split_partition_count: 2,
  }
  map_paged_prefill_capture_owner(
    table,
    profile.max_query_rows(),
    16,
    13,
    dispatch.prefill_split_slot_owner_indices,
  )
  map_paged_prefill_capture_owner(
    table,
    profile.max_query_rows(),
    32,
    14,
    dispatch.prefill_split_slot_owner_indices,
  )
  assert_eq(
    dispatch.select(graph_dispatch_bucket(table, PrefillGraph, 1, 12, 4096)),
    13,
  )
  assert_eq(
    dispatch.select(graph_dispatch_bucket(table, PrefillGraph, 1, 24, 4096)),
    14,
  )
  assert_eq(
    dispatch.select(graph_dispatch_bucket(table, PrefillGraph, 2, 24, 4096)),
    14,
  )
  assert_eq(
    dispatch.select(graph_dispatch_bucket(table, PrefillGraph, 1, 64, 4096)),
    15,
  )
  assert_eq(
    dispatch.select(graph_dispatch_bucket(table, PrefillGraph, 1, 12, 256)),
    7,
  )
}

///|
test "phase-specific decode owners replace baseline slots without touching prefill" {
  let (table, profile, baseline) = graph_dispatch_fixture(7)
  let dispatch : PagedOrderedExecutorDispatch = {
    measured_owner_indices: baseline.measured_owner_indices,
    slot_owner_indices: baseline.slot_owner_indices,
    decode_split_slot_owner_indices: baseline.decode_split_slot_owner_indices,
    prefill_wide_slot_owner_indices: baseline.prefill_wide_slot_owner_indices,
    prefill_split_slot_owner_indices: baseline.prefill_split_slot_owner_indices,
    fallback_owner_index: baseline.fallback_owner_index,
    telemetry_owner_index: baseline.telemetry_owner_index,
    prefill_wide_owner_index: baseline.prefill_wide_owner_index,
    prefill_wide_minimum_query_tokens: baseline.prefill_wide_minimum_query_tokens,
    prefill_split_owner_index: baseline.prefill_split_owner_index,
    prefill_split_minimum_context_tokens: baseline.prefill_split_minimum_context_tokens,
    prefill_split_query_tile: baseline.prefill_split_query_tile,
    prefill_split_partition_count: baseline.prefill_split_partition_count,
    deep_prefill_split_owner_index: baseline.deep_prefill_split_owner_index,
    deep_prefill_split_minimum_context_tokens: baseline.deep_prefill_split_minimum_context_tokens,
    deep_prefill_split_query_tile: baseline.deep_prefill_split_query_tile,
    deep_prefill_split_partition_count: baseline.deep_prefill_split_partition_count,
    decode_owner_index: 12,
    decode_split_owner_index: baseline.decode_split_owner_index,
    decode_split_minimum_context_tokens: baseline.decode_split_minimum_context_tokens,
    decode_split_maximum_batch_rows: baseline.decode_split_maximum_batch_rows,
  }
  let prefill = graph_dispatch_bucket(table, PrefillGraph, 32, 64, 1048576)
  assert_eq(dispatch.select(prefill), 2)
  map_paged_decode_capture_owner(table, 8, 8, 4, dispatch.slot_owner_indices)
  clear_paged_decode_capture_owners(
    table,
    profile.max_query_rows(),
    profile.max_query_rows(),
    dispatch.slot_owner_indices,
  )
  map_paged_decode_capture_owner(table, 1, 1, 8, dispatch.slot_owner_indices)
  map_paged_decode_capture_owner(table, 8, 8, 9, dispatch.slot_owner_indices)
  let one = graph_dispatch_bucket(table, DecodeGraph, 1, 1, 65536)
  let eight = graph_dispatch_bucket(table, DecodeGraph, 8, 8, 65536)
  let larger = graph_dispatch_bucket(table, DecodeGraph, 16, 16, 65536)
  assert_eq(dispatch.select(one), 8)
  assert_eq(dispatch.select(eight), 9)
  assert_eq(dispatch.select(larger), 12)
  assert_eq(dispatch.select(prefill), 2)
}

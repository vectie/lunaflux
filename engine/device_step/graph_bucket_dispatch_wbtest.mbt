///|
fn graph_dispatch_fixture(
  fallback_owner_index : Int,
) -> (
  @luna_execution_graph_strategy.ExecutionGraphBucketTable,
  @launch_contract.PagedKernelProfile,
  PagedOrderedExecutorDispatch,
) raise {
  let profile = @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(91),
    max_query_rows=32,
    max_query_tokens=64,
    max_page_table_entries=256,
  )
  let table = new_paged_graph_bucket_table(profile)
  let slot_owner_indices = FixedArray::make(table.bucket_count(), -1)
  for slot in paged_maximum_graph_slots(table, profile) {
    slot_owner_indices[slot] = 2
  }
  (
    table,
    profile,
    {
      slot_owner_indices,
      fallback_owner_index,
      telemetry_owner_index: 2,
      decode_owner_index: -1,
      decode_split_owner_index: -1,
      decode_split_minimum_context_tokens: PAGED_DECODE_SPLIT_DISABLED_CONTEXT_THRESHOLD,
    },
  )
}

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
  assert_eq(paged_prefill_capture_bounds(profile), [8, 16, 32])
  let larger = @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(92),
    max_query_rows=32,
    max_query_tokens=4096,
    max_page_table_entries=256,
  )
  assert_eq(paged_prefill_capture_bounds(larger), [
    8, 16, 32, 64, 128, 256, 512, 1024, 2048,
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
  assert_eq(dispatch.select(small), 5)
  assert_eq(dispatch.select(exact), 5)
  assert_eq(dispatch.select(larger), 7)
  assert_eq(dispatch.select(decode), 7)
}

///|
test "phase-specific decode owners replace baseline slots without touching prefill" {
  let (table, profile, baseline) = graph_dispatch_fixture(7)
  let dispatch : PagedOrderedExecutorDispatch = {
    slot_owner_indices: baseline.slot_owner_indices,
    fallback_owner_index: baseline.fallback_owner_index,
    telemetry_owner_index: baseline.telemetry_owner_index,
    decode_owner_index: 12,
    decode_split_owner_index: baseline.decode_split_owner_index,
    decode_split_minimum_context_tokens: baseline.decode_split_minimum_context_tokens,
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

///|
test "query-token launch scaling changes only grid x" {
  let source = @launch_contract.AotLaunchDimensions::new(
    grid_x=32,
    grid_y=14,
    grid_z=1,
    block_x=128,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
  let scaled = paged_scaled_query_token_launch(source, 8)
  assert_eq(
    (
      scaled.grid_x(),
      scaled.grid_y(),
      scaled.grid_z(),
      scaled.block_x(),
      scaled.block_y(),
      scaled.block_z(),
      scaled.shared_memory_bytes(),
    ),
    (8, 14, 1, 128, 1, 1, 0),
  )
}

///|
test "grid-stride query-token launch scales up to its admitted cap" {
  let source = @launch_contract.AotLaunchDimensions::new(
    grid_x=32,
    grid_y=16,
    grid_z=1,
    block_x=32,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
  assert_eq(paged_capped_query_token_launch(source, 8).grid_x(), 8)
  assert_eq(paged_capped_query_token_launch(source, 32).grid_x(), 32)
  assert_eq(paged_capped_query_token_launch(source, 256).grid_x(), 32)
}

///|
test "Qwen postprocess V2 caps its grid to the decode bucket" {
  let source = @launch_contract.AotLaunchDimensions::new(
    grid_x=256,
    grid_y=32,
    grid_z=1,
    block_x=128,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
  guard paged_qwen_postprocess_bucket_launch(source)
    is QueryTokenCappedGridX(admitted) else {
    fail("Qwen postprocess V2 did not expose a token-capped grid")
  }
  assert_eq(admitted, source)
  assert_eq(paged_bucket_launch(QueryTokenCappedGridX(admitted), 1).grid_x(), 1)
  assert_eq(paged_bucket_launch(QueryTokenCappedGridX(admitted), 8).grid_x(), 8)
}

///|
test "compiler attention buckets launch query tiles instead of token blocks" {
  let source = @launch_contract.AotLaunchDimensions::new(
    grid_x=496,
    grid_y=16,
    grid_z=1,
    block_x=256,
    block_y=1,
    block_z=1,
    shared_memory_bytes=31952,
  )
  let rule = QueryTileCappedGridX(source, 16)
  assert_eq(paged_bucket_launch(rule, 1).grid_x(), 1)
  assert_eq(paged_bucket_launch(rule, 128).grid_x(), 8)
  assert_eq(paged_bucket_launch(rule, 512).grid_x(), 32)
  assert_eq(paged_bucket_launch(rule, 4096).grid_x(), 256)
}

///|
test "phase-specific compiler decode scales one row per block" {
  let prefill = @launch_contract.AotLaunchDimensions::new(
    grid_x=256,
    grid_y=16,
    grid_z=1,
    block_x=256,
    block_y=1,
    block_z=1,
    shared_memory_bytes=31952,
  )
  let decode = @launch_contract.AotLaunchDimensions::new(
    grid_x=32,
    grid_y=8,
    grid_z=1,
    block_x=256,
    block_y=1,
    block_z=1,
    shared_memory_bytes=34076,
  )
  let prefill_scaled = paged_bucket_launch(QueryTileCappedGridX(prefill, 16), 8)
  let scaled = paged_bucket_launch(QueryTokenCappedGridX(decode), 8)
  assert_eq(prefill_scaled.grid_x(), 1)
  assert_eq(
    (scaled.grid_x(), scaled.grid_y(), scaled.shared_memory_bytes()),
    (8, 8, 34076),
  )
}

///|
test "projection buckets remove maximum-envelope decode blocks" {
  let dense = @launch_contract.AotLaunchDimensions::new(
    grid_x=4096,
    grid_y=1,
    grid_z=1,
    block_x=256,
    block_y=1,
    block_z=1,
    shared_memory_bytes=8192,
  )
  let dense_rule = ProjectionTensorCoreGridX(dense, 2048)
  assert_eq(paged_projection_bucket_grid_x(dense_rule, 1), 256)
  assert_eq(paged_projection_bucket_grid_x(dense_rule, 8), 16)
  assert_eq(paged_projection_bucket_grid_x(dense_rule, 32), 32)
  let scaled = paged_bucket_launch(dense_rule, 8)
  assert_eq(
    (scaled.grid_x(), scaled.block_x(), scaled.shared_memory_bytes()),
    (16, 256, 8192),
  )
}

///|
test "gated MLP buckets independently scale up and down projections" {
  let source = @launch_contract.AotLaunchDimensions::new(
    grid_x=3072,
    grid_y=1,
    grid_z=1,
    block_x=512,
    block_y=1,
    block_z=1,
    shared_memory_bytes=16384,
  )
  let up = ProjectionMlpUpGridX(source, 3072)
  let down = ProjectionMlpDownGridX(source, 1024)
  assert_eq(paged_projection_bucket_grid_x(up, 1), 192)
  assert_eq(paged_projection_bucket_grid_x(up, 8), 12)
  assert_eq(paged_projection_bucket_grid_x(up, 32), 24)
  assert_eq(paged_projection_bucket_grid_x(down, 1), 64)
  assert_eq(paged_projection_bucket_grid_x(down, 8), 16)
  assert_eq(paged_projection_bucket_grid_x(down, 32), 32)
}

///|
test "scalar projection bucket keeps only live output work" {
  let source = @launch_contract.AotLaunchDimensions::new(
    grid_x=16384,
    grid_y=1,
    grid_z=1,
    block_x=256,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
  let rule = ProjectionScalarGridX(source, 1024)
  assert_eq(paged_projection_bucket_grid_x(rule, 1), 4)
  assert_eq(paged_projection_bucket_grid_x(rule, 8), 32)
  assert_eq(paged_projection_bucket_grid_x(rule, 32), 128)
}

///|
fn graph_dispatch_bucket(
  table : @luna_execution_graph_strategy.ExecutionGraphBucketTable,
  phase : @luna_execution_graph_strategy.ExecutionGraphPhase,
  rows : Int,
  tokens : Int,
  context : Int,
) -> @luna_execution_graph_strategy.ExecutionGraphBucket raise {
  table.select(
    @luna_execution_graph_strategy.ExecutionGraphShape::new(
      phase~,
      batch_rows=rows,
      query_tokens=tokens,
      context_tokens=context,
    ),
  )
}

///|
test "sparse graph dispatch selects the shared exact max-envelope owner" {
  let (table, _, dispatch) = graph_dispatch_fixture(7)
  let phases : FixedArray[@luna_execution_graph_strategy.ExecutionGraphPhase] = [
    PrefillGraph,
    DecodeGraph,
    MixedGraph,
  ]
  for phase in phases {
    let bucket = graph_dispatch_bucket(
      table,
      phase,
      32,
      if phase == DecodeGraph {
        32
      } else {
        64
      },
      paged_graph_maximum_context_tokens(),
    )
    assert_eq(dispatch.select(bucket), 2)
  }
}

///|
test "sparse graph dispatch uses only the explicit eager fallback owner" {
  let (table, _, dispatch) = graph_dispatch_fixture(7)
  let bucket = graph_dispatch_bucket(table, DecodeGraph, 8, 1, 4096)
  assert_eq(dispatch.select(bucket), 7)
}

///|
test "missing graph bucket fails before descriptor staging without fallback" {
  let (table, _, dispatch) = graph_dispatch_fixture(-1)
  let bucket = graph_dispatch_bucket(table, PrefillGraph, 8, 8, 1024)
  try dispatch.select(bucket) catch {
    InvalidExecutor(GraphBucket) => ()
    _ => fail("unexpected missing-bucket failure")
  } noraise {
    _ => fail("missing bucket silently selected an executor")
  }
}

///|
test "required max-envelope capture preserves smaller-shape compatibility" {
  let (table, _, strict) = graph_dispatch_fixture(2)
  let bucket = graph_dispatch_bucket(table, MixedGraph, 4, 16, 2048)
  assert_eq(strict.select(bucket), 2)
}

///|
test "sparse executor cleanup retries only open owners in reverse order" {
  let open = [true, true, true]
  let fail_middle = Ref(true)
  let events : Array[Int] = []
  let close_once = () => {
    run_reverse_paged_executor_close(open.length(), index => {
      if open[index] {
        events.push(index)
        if index != 1 || !fail_middle.val {
          open[index] = false
        }
      }
    })
  }
  close_once()
  assert_eq(events, [2, 1, 0])
  assert_eq(open, [false, true, false])
  fail_middle.val = false
  close_once()
  assert_eq(events, [2, 1, 0, 1])
  assert_eq(open, [false, false, false])
}

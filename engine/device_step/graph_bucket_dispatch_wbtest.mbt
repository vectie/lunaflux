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
      decode_split_owner_index: -1,
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

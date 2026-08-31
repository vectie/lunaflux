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
  (
    table,
    profile,
    {
      exact_slots: paged_exact_maximum_graph_slots(table, profile),
      exact_slot_owner_indices: FixedArray::make(3, 2),
      fallback_owner_index,
      telemetry_owner_index: 2,
    },
  )
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
      64,
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

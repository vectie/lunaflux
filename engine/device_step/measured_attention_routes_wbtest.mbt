///|
test "measured runtime bundle binds routes and maps winners to real graph owners" {
  let plan = qwen_fused_test_plan()
  let blueprint = qwen_fused_test_blueprint(plan)
  let original = qwen_fused_test_runtime(
    plan,
    blueprint,
    phase_specific_attention=true,
    wide_query_prefill=true,
  )
  let scope = original.attention_route_scope()
  let bytes = @utf8.encode(
    "luna-attention-buckets-v1\nscope\t" +
    scope +
    "\nrecord\tprefill\t2\t4\t4096\t1\t100\t5\nrecord\tprefill\t2\t4\t4096\t2\t80\t5\n",
  )
  let runtime = qwen_fused_test_runtime(
    plan,
    blueprint,
    phase_specific_attention=true,
    wide_query_prefill=true,
    attention_bucket_routes=bytes,
  )
  let spans = admit_reusable_qwen_fused_attention_spans(
    plan,
    @worker_protocol.ModelPlanGeneration::first(),
    blueprint,
    runtime,
  )
  let candidates = qwen_measured_attention_candidates(spans).unwrap()
  let table = new_paged_graph_bucket_table(blueprint.profile())
  let bucket = graph_dispatch_bucket(table, PrefillGraph, 2, 4, 4096)
  assert_eq(candidates[bucket.slot()], 2)
  let base = FixedArray::make(candidates.length(), -1)
  let wide = FixedArray::make(candidates.length(), -1)
  wide[bucket.slot()] = 19
  let owners = map_measured_attention_owners(
    Some(candidates),
    base,
    7,
    wide,
    11,
  )
  assert_eq(owners[bucket.slot()], 19)
  assert_eq(
    owners[graph_dispatch_bucket(table, DecodeGraph, 2, 2, 4096).slot()],
    -1,
  )
  let wrong = @utf8.encode(@utf8.decode(bytes).replace(old=scope, new="wrong"))
  let stale = qwen_fused_test_runtime(
    plan,
    blueprint,
    phase_specific_attention=true,
    wide_query_prefill=true,
    attention_bucket_routes=wrong,
  )
  try
    admit_reusable_qwen_fused_attention_spans(
      plan,
      @worker_protocol.ModelPlanGeneration::first(),
      blueprint,
      stale,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("stale scope accepted")
  }
}

///|
test "decode and partitioned complete graphs use measured bucket owners" {
  let (table, _, base) = graph_dispatch_fixture(7)
  let ids = Array::make(table.bucket_count(), -1)
  let small = graph_dispatch_bucket(table, DecodeGraph, 1, 1, 4096)
  let large = graph_dispatch_bucket(table, DecodeGraph, 16, 16, 4096)
  ids[small.slot()] = 4
  ids[large.slot()] = 3
  let split = FixedArray::make(table.bucket_count(), -1)
  split[small.slot()] = 21
  let mapped = map_measured_attention_owners(
    Some(ReadOnlyArray::from_array(ids)),
    base.slot_owner_indices,
    7,
    base.prefill_wide_slot_owner_indices,
    11,
    decode_fallback=13,
    decode_split=Some(split),
    decode_split_fallback=23,
  )
  assert_eq(mapped[small.slot()], 21)
  assert_true(mapped[large.slot()] >= 0)
  try
    map_measured_attention_owners(
      Some(ReadOnlyArray::from_array(ids)),
      base.slot_owner_indices,
      7,
      base.prefill_wide_slot_owner_indices,
      11,
    )
  catch {
    InvalidExecutor(GraphBucket) => ()
    _ => fail("wrong unprepared route error")
  } noraise {
    _ => fail("unprepared measured partition route accepted")
  }
}

///|
test "measured baseline overrides heuristic while unmeasured buckets retain it" {
  let (table, _, base) = graph_dispatch_fixture(7)
  let chosen = graph_dispatch_bucket(table, PrefillGraph, 8, 64, 4096)
  let candidates = Array::make(table.bucket_count(), -1)
  candidates[chosen.slot()] = 1
  let dispatch : PagedOrderedExecutorDispatch = {
    ..base,
    prefill_wide_owner_index: 11,
    prefill_wide_minimum_query_tokens: 1,
    measured_owner_indices: map_measured_attention_owners(
      Some(ReadOnlyArray::from_array(candidates)),
      base.slot_owner_indices,
      7,
      base.prefill_wide_slot_owner_indices,
      11,
    ),
  }
  assert_eq(dispatch.select(chosen), 7)
  assert_eq(
    dispatch.select(graph_dispatch_bucket(table, PrefillGraph, 8, 64, 8192)),
    11,
  )
}

///|
test "same module decode crossover reaches startup owners and overrides split heuristics" {
  let (table, _, base) = graph_dispatch_fixture(7)
  let routes = @luna_attention_tuning.parse_attention_bucket_routes(
    @utf8.encode(
      "luna-attention-buckets-v1\nscope\tcomposed-module\n" +
      "record\tdecode\t8\t8\t128\t1\t10205\t5\nrecord\tdecode\t8\t8\t128\t4\t15662\t5\n" +
      "record\tdecode\t8\t8\t8192\t1\t1154628\t5\nrecord\tdecode\t8\t8\t8192\t4\t1174133\t5\n" +
      "record\tdecode\t1\t1\t8192\t1\t1000000\t5\nrecord\tdecode\t1\t1\t8192\t4\t500000\t5\n",
    ),
    scope="composed-module",
    table,
    baseline=1,
    [1, 4],
  )
  let split = FixedArray::make(table.bucket_count(), 23)
  let dispatch : PagedOrderedExecutorDispatch = {
    ..base,
    decode_split_owner_index: 23,
    decode_split_minimum_context_tokens: 1,
    decode_split_maximum_batch_rows: 32,
    measured_owner_indices: map_measured_attention_owners(
      Some(routes.bucket_candidates()),
      base.slot_owner_indices,
      7,
      base.prefill_wide_slot_owner_indices,
      11,
      decode_split=Some(split),
      decode_split_fallback=23,
    ),
  }
  for context in [128, 8192] {
    assert_eq(
      dispatch.select(graph_dispatch_bucket(table, DecodeGraph, 8, 8, context)),
      7,
    )
  }
  assert_eq(
    dispatch.select(graph_dispatch_bucket(table, DecodeGraph, 1, 1, 8192)),
    23,
  )
}

///|
test "mixed partitioned route requires its own prepared whole-graph owner" {
  let (table, _, base) = graph_dispatch_fixture(7)
  let bucket = graph_dispatch_bucket(table, MixedGraph, 2, 64, 8192)
  let ids = Array::make(table.bucket_count(), -1)
  ids[bucket.slot()] = 7
  let split = FixedArray::make(table.bucket_count(), -1)
  split[bucket.slot()] = 31
  let mapped = map_measured_attention_owners(
    Some(ReadOnlyArray::from_array(ids)),
    base.slot_owner_indices,
    7,
    base.prefill_wide_slot_owner_indices,
    11,
    mixed_decode_split=Some(split),
  )
  assert_eq(mapped[bucket.slot()], 31)
  let dispatch : PagedOrderedExecutorDispatch = {
    ..base,
    measured_owner_indices: mapped,
  }
  assert_eq(dispatch.select(bucket), 31)
  try
    map_measured_attention_owners(
      Some(ReadOnlyArray::from_array(ids)),
      base.slot_owner_indices,
      7,
      base.prefill_wide_slot_owner_indices,
      11,
      decode_split_fallback=23,
    )
  catch {
    InvalidExecutor(GraphBucket) => ()
    _ => fail("wrong missing mixed owner error")
  } noraise {
    _ => fail("pure decode owner substituted for mixed winner")
  }
}

///|
test "mixed partitioned measurements cannot be admitted as pure phase routes" {
  let plan = qwen_fused_test_plan()
  let blueprint = qwen_fused_test_blueprint(plan)
  let original = qwen_fused_test_runtime(
    plan,
    blueprint,
    phase_specific_attention=true,
    wide_query_prefill=true,
  )
  let scope = original.attention_route_scope()
  for phase in ["mixed", "prefill", "decode"] {
    let queries = if phase == "decode" { 2 } else { 4 }
    let runtime = qwen_fused_test_runtime(
      plan,
      blueprint,
      phase_specific_attention=true,
      wide_query_prefill=true,
      attention_bucket_routes=@utf8.encode(
        "luna-attention-buckets-v1\nscope\t" +
        scope +
        "\nrecord\t\{phase}\t2\t\{queries}\t4096\t1\t100\t5\nrecord\t\{phase}\t2\t\{queries}\t4096\t7\t80\t5\n",
      ),
    )
    if phase == "mixed" {
      let spans = admit_reusable_qwen_fused_attention_spans(
        plan,
        @worker_protocol.ModelPlanGeneration::first(),
        blueprint,
        runtime,
      )
      let table = new_paged_graph_bucket_table(blueprint.profile())
      let bucket = graph_dispatch_bucket(table, MixedGraph, 2, queries, 4096)
      assert_eq(
        qwen_measured_attention_candidates(spans).unwrap()[bucket.slot()],
        7,
      )
    } else {
      try
        admit_reusable_qwen_fused_attention_spans(
          plan,
          @worker_protocol.ModelPlanGeneration::first(),
          blueprint,
          runtime,
        )
      catch {
        _ => ()
      } noraise {
        _ => fail("mixed route admitted outside mixed phase")
      }
    }
  }
}

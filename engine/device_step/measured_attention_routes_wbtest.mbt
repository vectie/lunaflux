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

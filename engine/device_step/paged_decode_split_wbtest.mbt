///|
test "decode split phase selection never replaces prefill or mixed attention" {
  assert_true(
    paged_attention_phase_route(DecodeGraph, true) == DecodeSplitAttentionPhase,
  )
  assert_true(
    paged_attention_phase_route(PrefillGraph, true) == BaselineAttentionPhase,
  )
  assert_true(
    paged_attention_phase_route(MixedGraph, true) == BaselineAttentionPhase,
  )
  assert_true(
    paged_attention_phase_route(DecodeGraph, false) == BaselineAttentionPhase,
  )
}

///|
test "decode split dispatch owns only decode buckets" {
  let (table, _, base) = graph_dispatch_fixture(7)
  let dispatch : PagedOrderedExecutorDispatch = {
    slot_owner_indices: base.slot_owner_indices,
    fallback_owner_index: base.fallback_owner_index,
    telemetry_owner_index: base.telemetry_owner_index,
    decode_owner_index: 9,
    decode_split_owner_index: 11,
    decode_split_minimum_context_tokens: 2048,
  }
  assert_eq(
    dispatch.select(graph_dispatch_bucket(table, DecodeGraph, 8, 8, 1024)),
    9,
  )
  assert_eq(
    dispatch.select(graph_dispatch_bucket(table, DecodeGraph, 8, 8, 16384)),
    11,
  )
  assert_eq(
    dispatch.select(graph_dispatch_bucket(table, PrefillGraph, 8, 8, 4096)),
    7,
  )
  assert_eq(
    dispatch.select(graph_dispatch_bucket(table, MixedGraph, 8, 8, 4096)),
    7,
  )
}

///|
test "decode split workspace is one bounded startup envelope" {
  assert_eq(
    paged_decode_split_workspace_shape(
      ReadOnlyArray::from_array([(64L, 16L), (128L, 32L)]),
      256L,
    ),
    (128L, 32L),
  )
  try
    paged_decode_split_workspace_shape(
      ReadOnlyArray::from_array([(257L, 16L)]),
      256L,
    )
  catch {
    InvalidExecutor(ResourceState) => ()
    _ => fail("unexpected workspace-bound failure")
  } noraise {
    _ => fail("oversized decode split workspace was admitted")
  }
  try
    paged_decode_split_workspace_shape(
      ReadOnlyArray::from_array([(64L, 3L)]),
      256L,
    )
  catch {
    InvalidExecutor(ResourceState) => ()
    _ => fail("unexpected workspace-alignment failure")
  } noraise {
    _ => fail("invalid decode split alignment was admitted")
  }
}

///|
test "decode split replacement preserves partial then merge order" {
  let baseline = ReadOnlyArray::from_array([0, 1, 2, 3])
  let replaced = replace_paged_decode_split_steps(
    baseline,
    ReadOnlyArray::from_array([(2, 20, 21)]),
    value => value,
  )
  assert_eq(replaced, [0, 1, 20, 21, 3])
  try
    replace_paged_decode_split_steps(
      baseline,
      ReadOnlyArray::from_array([(9, 90, 91)]),
      value => value,
    )
  catch {
    InvalidExecutor(ResourceState) => ()
    _ => fail("unexpected missing-operation failure")
  } noraise {
    _ => fail("missing decode attention replacement was ignored")
  }
}

///|
test "attention allocation cleanup is explicit reverse acquisition order" {
  let events : Array[Int] = []
  run_paged_attention_allocation_close_order(
    () => events.push(4),
    () => events.push(3),
    () => events.push(2),
    () => events.push(1),
  )
  assert_eq(events, [4, 3, 2, 1])
}

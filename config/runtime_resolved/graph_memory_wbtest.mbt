///|
fn expect_graph_memory_error(
  availability : RuntimeGraphMemoryAvailability,
  bound : Int64?,
  alignment : Int64?,
  ceiling : Int64?,
  weight? : Int64 = 10L,
  activation? : Int64 = 20L,
  kv? : Int64 = 30L,
  expected~ : RuntimeCapacityRule,
) -> Unit raise {
  try
    resolve_graph_memory_parts(
      availability,
      bound,
      alignment,
      ceiling,
      weight_reserved_bytes=weight,
      activation_workspace_reserved_bytes=activation,
      kv_reserved_bytes=kv,
    )
  catch {
    Invalid(rule) => assert_eq(rule, expected)
  } noraise {
    _ => fail("expected graph-memory resolution rejection")
  }
}

///|
test "eager-only graph memory remains explicitly absent without zero" {
  for availability in [GraphAccountingAbsent, GraphAccountingUnsupported] {
    let resolved = resolve_graph_memory_parts(
      availability,
      None,
      None,
      None,
      weight_reserved_bytes=10L,
      activation_workspace_reserved_bytes=20L,
      kv_reserved_bytes=30L,
    )
    assert_eq(resolved.availability(), GraphAccountingAbsent)
    assert_eq(resolved.arena_reserved_total_bytes(), 60L)
    assert_eq(resolved.declared_upper_bound_bytes(), None)
    assert_eq(resolved.descriptor_ceiling_bytes(), None)
    assert_eq(resolved.device_startup_upper_bound_bytes(), None)
  }
  let inactive_bound = resolve_graph_memory_parts(
    GraphAccountingBounded,
    Some(4096L),
    Some(256L),
    None,
    weight_reserved_bytes=10L,
    activation_workspace_reserved_bytes=20L,
    kv_reserved_bytes=30L,
  )
  assert_eq(inactive_bound.availability(), GraphAccountingAbsent)
  assert_eq(inactive_bound.declared_upper_bound_bytes(), None)
  assert_eq(inactive_bound.descriptor_ceiling_bytes(), None)
  assert_eq(inactive_bound.device_startup_upper_bound_bytes(), None)
}

///|
test "bounded graph memory is added once beneath its descriptor ceiling" {
  let resolved = resolve_graph_memory_parts(
    GraphAccountingBounded,
    Some(4096L),
    Some(256L),
    Some(8192L),
    weight_reserved_bytes=10L,
    activation_workspace_reserved_bytes=20L,
    kv_reserved_bytes=30L,
  )
  assert_eq(resolved.availability(), GraphAccountingBounded)
  assert_eq(resolved.arena_reserved_total_bytes(), 60L)
  assert_eq(resolved.declared_upper_bound_bytes(), Some(4096L))
  assert_eq(resolved.declared_alignment_bytes(), Some(256L))
  assert_eq(resolved.descriptor_ceiling_bytes(), Some(8192L))
  assert_eq(resolved.device_startup_upper_bound_bytes(), Some(4156L))
}

///|
test "graph memory rejects ceiling underflow mismatch and overflow" {
  expect_graph_memory_error(
    GraphAccountingBounded,
    Some(4096L),
    Some(256L),
    Some(2048L),
    expected=GraphMemoryCeiling,
  )
  expect_graph_memory_error(
    GraphAccountingBounded,
    Some(4097L),
    Some(256L),
    Some(8192L),
    expected=GraphMemoryCeiling,
  )
  expect_graph_memory_error(
    GraphAccountingAbsent,
    None,
    None,
    Some(8192L),
    expected=GraphMemoryEvidence,
  )
  expect_graph_memory_error(
    GraphAccountingUnsupported,
    None,
    None,
    Some(8192L),
    expected=GraphMemoryEvidence,
  )
  expect_graph_memory_error(
    GraphAccountingBounded,
    Some(4096L),
    Some(256L),
    Some(8192L),
    weight=0x7fffffffffffffffL,
    activation=1L,
    expected=DeviceMemoryUpperBound,
  )
  expect_graph_memory_error(
    GraphAccountingBounded,
    Some(4096L),
    Some(256L),
    Some(8192L),
    weight=0x7ffffffffffff000L,
    activation=0L,
    kv=0L,
    expected=DeviceMemoryUpperBound,
  )
}

///|
fn arena_test_identity(seed : String) -> @spec.ModelIdentity raise {
  @spec.ModelIdentity::new(
    @spec.ContentDigest::from_sha256(seed),
    @spec.PlanDigest::from_sha256(seed),
  )
}

///|
fn arena_test_layout(
  identity : @spec.ModelIdentity,
) -> @device_layout.DeviceKvLayout raise {
  @device_layout.plan(
    identity,
    @plan.KvCacheGeometry::new(
      layer_count=2,
      kv_head_count=2,
      head_dimension=4,
      dtype=BFloat16,
      layout=LayerMajorSplitKeyValueV1,
    ),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=4,
      total_page_count=3,
      max_arena_bytes=3072L,
    ),
  )
}

///|
test "preflight retains the exact immutable KV envelope" {
  let identity = arena_test_identity(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  let plan = @device_arena.preflight(identity, arena_test_layout(identity))
  assert_eq(plan.identity(), identity)
  assert_eq(plan.arena_bytes(), 3072L)
  assert_eq(plan.segment_alignment_bytes(), 256L)
  assert_eq(plan.layout().page_logical_bytes(), 64L)
  assert_eq(plan.layout().page_stride_bytes(), 256L)
  assert_eq(plan.layout().component_stride_bytes(), 768L)
  assert_eq(plan.layout().layer_stride_bytes(), 1536L)
}

///|
test "preflight rejects a mismatched model generation" {
  let layout_identity = arena_test_identity(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  let expected_identity = arena_test_identity(
    "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
  )
  try
    @device_arena.preflight(
      expected_identity,
      arena_test_layout(layout_identity),
    )
  catch {
    @device_arena.InvalidLayout(Identity) => ()
    _ => fail("unexpected identity failure")
  } noraise {
    _ => fail("expected identity mismatch")
  }
}

///|
test "preflight proves exact first and last canonical regions" {
  let identity = arena_test_identity(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  let plan = @device_arena.preflight(identity, arena_test_layout(identity))
  let first = plan.layout().region(0, Key, 0)
  let last = plan.layout().region(1, Value, 2)
  assert_eq(first.offset_bytes(), 0L)
  assert_eq(first.end_offset_bytes(), 256L)
  assert_eq(last.offset_bytes(), 2816L)
  assert_eq(last.end_offset_bytes(), 3072L)
}

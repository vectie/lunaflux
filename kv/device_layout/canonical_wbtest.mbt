///|
test "canonical re-admission rejects forged derived scalars" {
  let identity = @spec.ModelIdentity::new(
    @spec.ContentDigest::from_sha256(
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
    @spec.PlanDigest::from_sha256(
      "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
    ),
  )
  let geometry = @plan.KvCacheGeometry::new(
    layer_count=1,
    kv_head_count=1,
    head_dimension=4,
    dtype=BFloat16,
    layout=LayerMajorSplitKeyValueV1,
  )
  let layout = plan(
    identity,
    geometry,
    DeviceKvLimits::new(
      tokens_per_page=4,
      total_page_count=2,
      max_arena_bytes=1024L,
    ),
  )
  assert_true(layout.is_canonical_for(identity, geometry))
  let forged = { ..layout, arena_bytes: layout.arena_bytes() + 256L }
  assert_false(forged.is_canonical_for(identity, geometry))
}

///|
fn layout_identity() -> @spec.ModelIdentity {
  @spec.ModelIdentity::new(
    try! @spec.ContentDigest::from_sha256(
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
    try! @spec.PlanDigest::from_sha256(
      "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
    ),
  )
}

///|
fn layout_geometry(
  layers : Int,
  heads : Int,
  width : Int,
) -> @plan.KvCacheGeometry {
  try! @plan.KvCacheGeometry::new(
    layer_count=layers,
    kv_head_count=heads,
    head_dimension=width,
    dtype=BFloat16,
    layout=LayerMajorSplitKeyValueV1,
  )
}

///|
fn layout_limits(
  tokens : Int,
  pages : Int,
  maximum : Int64,
) -> @device_layout.DeviceKvLimits {
  try! @device_layout.DeviceKvLimits::new(
    tokens_per_page=tokens,
    total_page_count=pages,
    max_arena_bytes=maximum,
  )
}

///|
test "canonical two-layer fixture occupies exactly 3072 bytes" {
  let identity = layout_identity()
  let geometry = layout_geometry(2, 2, 4)
  let limits = layout_limits(4, 3, 3072L)
  let layout = @device_layout.plan(identity, geometry, limits)
  assert_eq(layout.identity(), identity)
  assert_eq(layout.geometry(), geometry)
  assert_eq(layout.limits(), limits)
  assert_eq(layout.stable_layout_version(), 1)
  assert_eq(layout.tokens_per_page(), 4)
  assert_eq(layout.total_page_count(), 3)
  assert_eq(layout.segment_alignment_bytes(), 256L)
  assert_eq(layout.page_logical_bytes(), 64L)
  assert_eq(layout.page_stride_bytes(), 256L)
  assert_eq(layout.component_stride_bytes(), 768L)
  assert_eq(layout.layer_stride_bytes(), 1536L)
  assert_eq(layout.arena_bytes(), 3072L)

  let key_zero = layout.region(0, Key, 0)
  assert_eq(key_zero.offset_bytes(), 0L)
  assert_eq(key_zero.logical_bytes(), 64L)
  assert_eq(key_zero.reserved_bytes(), 256L)
  assert_eq(key_zero.alignment_bytes(), 256L)
  assert_eq(key_zero.end_offset_bytes(), 256L)
  assert_eq(key_zero.layer_index(), 0)
  assert_eq(key_zero.component(), Key)
  assert_eq(key_zero.page_index(), 0)
  assert_eq(layout.region(0, Key, 2).offset_bytes(), 512L)
  assert_eq(layout.region(0, Value, 0).offset_bytes(), 768L)
  assert_eq(layout.region(1, Key, 0).offset_bytes(), 1536L)
  assert_eq(layout.region(1, Value, 2).offset_bytes(), 2816L)
  assert_eq(layout.region(1, Value, 2).end_offset_bytes(), 3072L)
  assert_eq(layout.element_offset_bytes(0, Key, 0, 0, 0, 0), 0L)
  assert_eq(layout.element_offset_bytes(0, Key, 0, 1, 0, 0), 16L)
  assert_eq(layout.element_offset_bytes(0, Key, 0, 3, 1, 3), 62L)
  assert_eq(layout.element_offset_bytes(1, Value, 2, 3, 1, 3), 2878L)
}

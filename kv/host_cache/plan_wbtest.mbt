///|
fn fixture_layout() -> @device_layout.DeviceKvLayout raise {
  let digest = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  @device_layout.plan(
    @spec.ModelIdentity::new(
      @spec.ContentDigest::from_sha256(digest),
      @spec.PlanDigest::from_sha256(digest),
    ),
    @plan.KvCacheGeometry::new(
      layer_count=3,
      kv_head_count=2,
      head_dimension=4,
      dtype=BFloat16,
      layout=LayerMajorSplitKeyValueV1,
    ),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=4,
      total_page_count=7,
      max_arena_bytes=10752L,
    ),
  )
}

///|
test "pure streaming mapping exactly matches every canonical KV region" {
  let layout = fixture_layout()
  let plan = Plan::new(
    layout,
    slots=5,
    transfer_lanes=2,
    host_budget=1920L,
    separate_host_memory=true,
  )
  assert_eq(plan.slot_bytes(), 384L)
  assert_eq(plan.host_bytes(), 1920L)
  for slot in 0..<plan.slots() {
    for page in 0..<layout.total_page_count() {
      for layer in 0..<3 {
        for component in 0..<2 {
          let copy = plan.region(slot, page, layer * 2 + component)
          let reference = layout.region(
            layer,
            if component == 0 {
              Key
            } else {
              Value
            },
            page,
          )
          assert_eq(copy.device_offset, reference.offset_bytes())
          assert_eq(copy.bytes, reference.logical_bytes())
          assert_eq(
            copy.host_offset,
            slot.to_int64() * 384L + (layer * 2 + component).to_int64() * 64L,
          )
          assert_true(copy.host_offset + copy.bytes <= plan.host_bytes())
          assert_true(copy.device_offset + copy.bytes <= layout.arena_bytes())
        }
      }
    }
  }
}

///|
test "host capacity and shared-memory rejection happen before effects" {
  let layout = fixture_layout()
  try
    Plan::new(
      layout,
      slots=5,
      transfer_lanes=2,
      host_budget=1919L,
      separate_host_memory=true,
    )
  catch {
    @residency.InvalidLimits => ()
  } noraise {
    _ => fail("budget")
  }
  try
    Plan::new(
      layout,
      slots=5,
      transfer_lanes=2,
      host_budget=1920L,
      separate_host_memory=false,
    )
  catch {
    @residency.InvalidLimits => ()
  } noraise {
    _ => fail("shared memory capacity")
  }
}

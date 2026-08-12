///|
fn executor_test_kv_geometry() -> @plan.KvCacheGeometry {
  try! @plan.KvCacheGeometry::new(
    layer_count=1,
    kv_head_count=1,
    head_dimension=2,
    dtype=BFloat16,
    layout=LayerMajorSplitKeyValueV1,
  )
}

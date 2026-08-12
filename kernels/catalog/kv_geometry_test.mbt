///|
fn catalog_test_kv_geometry() -> @plan.KvCacheGeometry {
  try! @plan.KvCacheGeometry::new(
    layer_count=1,
    kv_head_count=2,
    head_dimension=4,
    dtype=BFloat16,
    layout=LayerMajorSplitKeyValueV1,
  )
}

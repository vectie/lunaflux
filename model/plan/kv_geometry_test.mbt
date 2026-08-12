///|
test "KV geometry has immutable canonical BF16 semantic identity" {
  let source_layer_count = Ref(32)
  let geometry = @plan.KvCacheGeometry::new(
    layer_count=source_layer_count.val,
    kv_head_count=8,
    head_dimension=128,
    dtype=BFloat16,
    layout=LayerMajorSplitKeyValueV1,
  )
  let same_geometry = @plan.KvCacheGeometry::new(
    layer_count=32,
    kv_head_count=8,
    head_dimension=128,
    dtype=BFloat16,
    layout=LayerMajorSplitKeyValueV1,
  )
  let other_geometry = @plan.KvCacheGeometry::new(
    layer_count=31,
    kv_head_count=8,
    head_dimension=128,
    dtype=BFloat16,
    layout=LayerMajorSplitKeyValueV1,
  )
  source_layer_count.val = 1
  assert_eq(geometry.layer_count(), 32)
  assert_eq(geometry.kv_head_count(), 8)
  assert_eq(geometry.head_dimension(), 128)
  assert_eq(geometry.dtype(), BFloat16)
  assert_eq(geometry.layout(), LayerMajorSplitKeyValueV1)
  assert_eq(geometry.layout().stable_version(), 1)
  assert_eq(geometry, same_geometry)
  assert_false(geometry == other_geometry)
}

///|
test "KV geometry rejects every nonpositive dimension" {
  let dimensions = [(0, 2, 4), (1, 0, 4), (1, 2, 0), (-1, 2, 4)]
  for dimensions_case in dimensions {
    let (layers, heads, width) = dimensions_case
    try
      @plan.KvCacheGeometry::new(
        layer_count=layers,
        kv_head_count=heads,
        head_dimension=width,
        dtype=BFloat16,
        layout=LayerMajorSplitKeyValueV1,
      )
    catch {
      InvalidShape(KvCachePositiveDimension) => ()
      _ => fail("unexpected KV dimension error")
    } noraise {
      _ => fail("expected nonpositive KV dimension rejection")
    }
  }
}

///|
test "KV geometry rejects a nonrepresentable per-token width" {
  try
    @plan.KvCacheGeometry::new(
      layer_count=1,
      kv_head_count=1_073_741_824,
      head_dimension=2,
      dtype=BFloat16,
      layout=LayerMajorSplitKeyValueV1,
    )
  catch {
    InvalidShape(KvCacheDimensionOverflow) => ()
    _ => fail("unexpected KV overflow error")
  } noraise {
    _ => fail("expected KV width overflow rejection")
  }
}

///|
test "KV geometry rejects unsupported persistent storage precision" {
  let dtypes : Array[@spec.ModelDType] = [Float16, Float32]
  for dtype in dtypes {
    try
      @plan.KvCacheGeometry::new(
        layer_count=1,
        kv_head_count=2,
        head_dimension=4,
        dtype~,
        layout=LayerMajorSplitKeyValueV1,
      )
    catch {
      InvalidShape(UnsupportedKvCacheDType) => ()
      _ => fail("unexpected KV dtype error")
    } noraise {
      _ => fail("expected unsupported KV dtype rejection")
    }
  }
}

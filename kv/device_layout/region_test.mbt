///|
test "region lookup rejects negative and terminal indices" {
  let layout = @device_layout.plan(
    layout_identity(),
    layout_geometry(2, 2, 4),
    layout_limits(4, 3, 3072L),
  )
  for index in [-1, 2] {
    try layout.region(index, Key, 0) catch {
      IndexOutOfBounds(Layer) => ()
      _ => fail("unexpected layer-index error")
    } noraise {
      _ => fail("expected layer-index rejection")
    }
  }
  for index in [-1, 3] {
    try layout.region(0, Value, index) catch {
      IndexOutOfBounds(Page) => ()
      _ => fail("unexpected page-index error")
    } noraise {
      _ => fail("expected page-index rejection")
    }
  }
}

///|
test "element lookup rejects every scalar dimension boundary" {
  let layout = @device_layout.plan(
    layout_identity(),
    layout_geometry(2, 2, 4),
    layout_limits(4, 3, 3072L),
  )
  let cases : Array[(Int, Int, Int, @device_layout.DeviceKvIndexKind)] = [
    (-1, 0, 0, Token),
    (4, 0, 0, Token),
    (0, -1, 0, KvHead),
    (0, 2, 0, KvHead),
    (0, 0, -1, HeadDimension),
    (0, 0, 4, HeadDimension),
  ]
  for case in cases {
    let (token, head, dimension, expected) = case
    try layout.element_offset_bytes(0, Key, 0, token, head, dimension) catch {
      IndexOutOfBounds(actual) if actual == expected => ()
      _ => fail("unexpected device-KV element-index error")
    } noraise {
      _ => fail("expected device-KV element-index rejection")
    }
  }
}

///|
fn next_layout_seed(seed : UInt64) -> UInt64 {
  seed * 6364136223846793005UL + 1442695040888963407UL
}

///|
test "randomized layouts are aligned contiguous and nonoverlapping" {
  let mut seed = 0x13579bdf2468ace1UL
  for _ in 0..<256 {
    seed = next_layout_seed(seed)
    let layers = (seed % 8UL).to_int() + 1
    seed = next_layout_seed(seed)
    let heads = (seed % 8UL).to_int() + 1
    seed = next_layout_seed(seed)
    let width = (seed % 16UL).to_int() + 1
    seed = next_layout_seed(seed)
    let tokens = (seed % 32UL).to_int() + 1
    seed = next_layout_seed(seed)
    let pages = (seed % 16UL).to_int() + 1
    let geometry = layout_geometry(layers, heads, width)
    let layout = @device_layout.plan(
      layout_identity(),
      geometry,
      layout_limits(tokens, pages, 1073741824L),
    )
    let expected_logical = Int64::from_int(tokens) *
      Int64::from_int(heads) *
      Int64::from_int(width) *
      2L
    let expected_stride = (expected_logical + 255L) / 256L * 256L
    assert_eq(layout.page_logical_bytes(), expected_logical)
    assert_eq(layout.page_stride_bytes(), expected_stride)
    assert_eq(
      layout.component_stride_bytes(),
      expected_stride * Int64::from_int(pages),
    )
    assert_eq(layout.layer_stride_bytes(), layout.component_stride_bytes() * 2L)
    assert_eq(
      layout.arena_bytes(),
      layout.layer_stride_bytes() * Int64::from_int(layers),
    )
    let mut expected_offset = 0L
    for layer in 0..<layers {
      let components : Array[@device_layout.DeviceKvComponent] = [Key, Value]
      for component in components {
        for page in 0..<pages {
          let region = layout.region(layer, component, page)
          assert_eq(region.offset_bytes(), expected_offset)
          assert_eq(region.offset_bytes() % 256L, 0L)
          assert_true(region.logical_bytes() <= region.reserved_bytes())
          assert_eq(
            region.end_offset_bytes(),
            region.offset_bytes() + layout.page_stride_bytes(),
          )
          assert_eq(
            layout.element_offset_bytes(
              layer,
              component,
              page,
              tokens - 1,
              heads - 1,
              width - 1,
            ),
            region.offset_bytes() + region.logical_bytes() - 2L,
          )
          expected_offset = region.end_offset_bytes()
        }
      }
    }
    assert_eq(expected_offset, layout.arena_bytes())
  }
}

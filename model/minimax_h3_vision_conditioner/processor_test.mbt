///|
fn pixel_value(
  p : ProcessedPixels,
  patch : Int,
  c : Int,
  t : Int,
  y : Int,
  x : Int,
) -> UInt {
  let i = (patch * 1536 + c * 512 + t * 256 + y * 16 + x) * 2
  p.patches[i].to_uint() | (p.patches[i + 1].to_uint() << 8)
}

///|
test "RGB processor duplicates image temporal slot and groups spatial patches" {
  let data : Array[Byte] = Array::makei(32 * 32 * 3, i => {
    let c = i % 3
    let y = i / (32 * 3)
    let x = i / 3 % 32
    if c == 0 {
      255
    } else if c == 1 {
      0
    } else if x >= 16 && y >= 16 {
      255
    } else {
      0
    }
  })
  let p = preprocess_rgb(
    [Bytes::from_array(data)],
    height=32,
    width=32,
    min_pixels=1024,
    max_pixels=1024,
    max_bytes=100000L,
  )
  assert_true(p.grid == { time: 1, height: 2, width: 2, })
  assert_eq(p.patches.length(), 4 * 1536 * 2)
  for patch in 0..<4 {
    for t in 0..<2 {
      assert_eq(pixel_value(p, patch, 0, t, 0, 0), 0x3f80U)
      assert_eq(pixel_value(p, patch, 1, t, 0, 0), 0xbf80U)
      assert_eq(
        pixel_value(p, patch, 2, t, 0, 0),
        if patch == 3 {
          0x3f80U
        } else {
          0xbf80U
        },
      )
    }
  }
}

///|
test "RGB processor pads odd videos without mixing temporal pairs" {
  let dark = Bytes::from_array(Array::make(32 * 32 * 3, 0))
  let light = Bytes::from_array(Array::make(32 * 32 * 3, 255))
  let p = preprocess_rgb(
    [dark, light, dark],
    height=32,
    width=32,
    min_pixels=1024,
    max_pixels=1024,
    max_bytes=100000L,
  )
  assert_eq(p.grid.time, 2)
  assert_eq(pixel_value(p, 0, 0, 0, 0, 0), 0xbf80U)
  assert_eq(pixel_value(p, 0, 0, 1, 0, 0), 0x3f80U)
  assert_eq(pixel_value(p, 4, 0, 0, 0, 0), 0xbf80U)
  assert_eq(pixel_value(p, 4, 0, 1, 0, 0), 0xbf80U)
}

///|
test "RGB smart resize bounds and constant resampling" {
  assert_eq(resized_shape(48, 48, min_pixels=1024, max_pixels=100000), (64, 64))
  assert_eq(resized_shape(80, 80, min_pixels=1024, max_pixels=100000), (64, 64))
  let frame = Bytes::from_array(Array::make(16 * 16 * 3, 255))
  let p = preprocess_rgb(
    [frame],
    height=16,
    width=16,
    min_pixels=1024,
    max_pixels=1024,
    max_bytes=100000L,
  )
  assert_eq(pixel_value(p, 3, 2, 1, 15, 15), 0x3f80U)
  try
    preprocess_rgb(
      [frame],
      height=16,
      width=16,
      min_pixels=1024,
      max_pixels=1024,
      max_bytes=1L,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("unbounded preprocess accepted")
  }
  try resized_shape(1, 1000) catch {
    _ => ()
  } noraise {
    _ => fail("extreme aspect accepted")
  }
}

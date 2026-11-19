///|
async test "PNG RGB and alpha conversion retain stored channel words" {
  for name in ["rgb", "rgba"] {
    let image = @rgb8.Image::decode(
      @fs.read_file("media/rgb8/fixtures/" + name + ".png").binary(),
      maximum_encoded_bytes=4096,
      maximum_pixels=2,
    )
    assert_eq((image.width(), image.height()), (2, 1))
    assert_eq(image.data(), b"\xff\x00\x00\x00\xff\x00")
  }
  let gray = @rgb8.Image::decode(
    @fs.read_file("media/rgb8/fixtures/gray.png").binary(),
    maximum_encoded_bytes=4096,
    maximum_pixels=2,
  )
  assert_eq(gray.data(), b"\x00\x00\x00\x80\x80\x80")
}

///|
async test "baseline and progressive JPEG decode to bounded RGB8" {
  for name in ["baseline", "progressive"] {
    let image = @rgb8.Image::decode(
      @fs.read_file("media/rgb8/fixtures/" + name + ".jpg").binary(),
      maximum_encoded_bytes=4096,
      maximum_pixels=64,
    )
    assert_eq((image.width(), image.height()), (8, 8))
    for i in 0..<64 {
      assert_true(image.data()[i * 3].to_int() >= 253)
      assert_true(image.data()[i * 3 + 1].to_int() <= 1)
      assert_true(image.data()[i * 3 + 2].to_int() <= 1)
    }
  }
}

///|
async test "decode rejects invalid encoding precision and size ceilings" {
  let valid = @fs.read_file("media/rgb8/fixtures/rgb.png").binary()
  assert_true(
    try
      @rgb8.Image::decode(valid, maximum_encoded_bytes=4096, maximum_pixels=1)
    catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  assert_true(
    try
      @rgb8.Image::decode(valid, maximum_encoded_bytes=1, maximum_pixels=2)
    catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  assert_true(
    try
      @rgb8.Image::decode(valid, maximum_encoded_bytes=4096, maximum_pixels=0)
    catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  assert_true(
    try
      @rgb8.Image::decode(
        @fs.read_file("media/rgb8/fixtures/rgb16.png").binary(),
        maximum_encoded_bytes=4096,
        maximum_pixels=2,
      )
    catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  for bytes in [b"", b"not an image", valid[:30].to_owned()] {
    assert_true(
      try
        @rgb8.Image::decode(bytes, maximum_encoded_bytes=4096, maximum_pixels=2)
      catch {
        _ => true
      } noraise {
        _ => false
      },
    )
  }
}

///|
async test "progressive JPEG retains entropy buffered before the next marker" {
  let image = @rgb8.Image::decode(
    @fs.read_file("media/rgb8/fixtures/progressive-detail.jpg").binary(),
    maximum_encoded_bytes=4096,
    maximum_pixels=187,
  )
  let expected = @fs.read_file("media/rgb8/fixtures/progressive-detail.rgb").binary()
  assert_eq((image.width(), image.height(), expected.length()), (17, 11, 561))
  for i in 0..<expected.length() {
    assert_true((image.data()[i].to_int() - expected[i].to_int()).abs() <= 2)
  }
}

///|
test "RGB frame constructor rejects extent mismatches" {
  assert_true(
    try @rgb8.Image::from_rgb(b"", width=1, height=1, maximum_pixels=1) catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  assert_true(
    try
      @rgb8.Image::from_rgb(
        b"\x00\x00\x00",
        width=-1,
        height=1,
        maximum_pixels=1,
      )
    catch {
      _ => true
    } noraise {
      _ => false
    },
  )
}

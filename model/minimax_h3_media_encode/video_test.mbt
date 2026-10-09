///|
test "video complete tile schedule covers canvas with aligned overlap" {
  for size in [32, 256, 288, 768, 1344] {
    let axis = @minimax_h3_media_encode.TileAxis::new(size)
    assert_eq(axis.starts[0], 0)
    assert_eq(axis.starts[axis.starts.length() - 1] + axis.extent, size)
    assert_eq(axis.overlaps.length(), axis.starts.length() - 1)
    for i, overlap in axis.overlaps {
      assert_true(overlap >= 64 && overlap % 16 == 0)
      assert_eq(axis.starts[i + 1], axis.starts[i] + axis.extent - overlap)
    }
  }
}

///|
test "image encoder includes all twelve residual blocks and actual posterior" {
  let video = @minimax_h3_media_encode.VideoEncoder::new(
    frames=1,
    height=32,
    width=32,
    image=true,
    max_tiles=1,
    max_scratch_bytes=2000000000L,
  )
  assert_eq(video.latent_time, 1)
  assert_eq(video.tile_count, 1)
  assert_eq(video.graph.values[video.graph.output].elements, 96L)
  let mut convolution = 0
  let mut normalization = 0
  let mut residual = 0
  let mut posterior = 0
  for node in video.graph.nodes {
    match node.operation {
      Conv3d(..) => convolution += 1
      GroupNormSilu(..) => normalization += 1
      Add(..) => residual += 1
      GaussianPatch(..) => posterior += 1
      _ => ()
    }
  }
  assert_eq(convolution, 34)
  assert_eq(normalization, 25)
  assert_eq(residual, 12)
  assert_eq(posterior, 1)
}

///|
test "video temporal clip recipe drops three latent cells only at final tail" {
  for (frames, expected) in [(1, 2), (17, 2), (18, 7), (107, 32), (125, 37)] {
    let video = @minimax_h3_media_encode.VideoEncoder::new(
      frames~,
      height=32,
      width=32,
      image=false,
      max_tiles=8,
      max_scratch_bytes=2000000000L,
    )
    assert_eq(video.latent_time, expected)
    assert_eq(
      video.graph.values[video.noise_value].elements,
      24L * expected.to_int64() * 4L,
    )
  }
  try
    @minimax_h3_media_encode.VideoEncoder::new(
      frames=18,
      height=32,
      width=32,
      image=false,
      max_tiles=1,
      max_scratch_bytes=2000000000L,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("accepted tile budget overrun")
  }
}

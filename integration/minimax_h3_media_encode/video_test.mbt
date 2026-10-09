///|
test "video source has causal CNN temporal isolated GN and sampled half latent" {
  let source = @minimax_h3_media_encode.VideoSource::new(
    frames=1,
    height=32,
    width=32,
    image=true,
    max_tiles=1,
    max_scratch_bytes=2000000000L,
  )
  assert_eq(source.kernels.symbols.length(), source.plan.graph.nodes.length())
  let text = @utf8.decode(source.kernels.source)
  assert_true(text.contains("ti<0"))
  assert_true(text.contains("1e-6f"))
  assert_true(text.contains("__float2half_rn(sampled)"))
  assert_true(text.contains("fmaxf(-30.0f"))
  assert_false(text.contains("curand"))
}

///|
test "RGB conversion is chunk independent and channels first" {
  let rgb : Bytes = [255, 0, 0, 0, 255, 0]
  let full = @minimax_h3_media_encode.rgb_chunk(
    rgb,
    start_element=0,
    count=6,
    frames=1,
    height=1,
    width=2,
  )
  let left = @minimax_h3_media_encode.rgb_chunk(
    rgb,
    start_element=0,
    count=3,
    frames=1,
    height=1,
    width=2,
  )
  let right = @minimax_h3_media_encode.rgb_chunk(
    rgb,
    start_element=3,
    count=3,
    frames=1,
    height=1,
    width=2,
  )
  assert_eq(full[:12].to_bytes(), left)
  assert_eq(full[12:].to_bytes(), right)
  let expected = ((1.0F - 0.485F) / 0.229F).reinterpret_as_uint()
  for i in 0..<4 {
    assert_eq(full[i], ((expected >> (i * 8)) & 255U).to_byte())
  }
  try
    @minimax_h3_media_encode.rgb_chunk(
      rgb,
      start_element=5,
      count=2,
      frames=1,
      height=1,
      width=2,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("accepted outside RGB input")
  }
}

///|
test "video request refuses execution and output before prepare" {
  let request = @minimax_h3_media_encode.VideoRequest::new()
  try request.execute() catch {
    _ => ()
  } noraise {
    _ => fail("executed empty request")
  }
  try request.with_destination((_, _, _) => ()) catch {
    _ => ()
  } noraise {
    _ => fail("borrowed absent destination")
  }
  try request.with_rows((_, _, _) => ()) catch {
    _ => ()
  } noraise {
    _ => fail("read absent output")
  }
  request.cancel()
  request.close()
  request.close()
}

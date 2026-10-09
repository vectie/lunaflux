///|
test "canonical output preserves inverse normalization ordering and crop strides" {
  let plan = @minimax_h3_video_decode.VideoOutput::new(
    frames=22,
    input_height=64,
    input_width=80,
    height=60,
    width=72,
    symbol="canonical",
  )
  assert_eq(plan.output_bytes, 22L * 60L * 72L * 12L)
  assert_eq(plan.input_bytes, 22L * 64L * 80L * 12L)
  let source = @utf8.decode(plan.source)
  assert_true(source.contains("__fdiv_rn(__fsub_rn("))
  assert_true(source.contains("*64ULL+y)*80ULL+x"))
  assert_true(source.contains("i%72ULL"))
  assert_true(source.contains("v<0.0f?0.0f:(v>1.0f?1.0f:v)"))
  assert_false(source.contains("fmaf"))
  assert_false(source.contains("fminf"))
  try
    @minimax_h3_video_decode.VideoOutput::new(
      frames=22,
      input_height=64,
      input_width=80,
      height=65,
      width=72,
      symbol="canonical",
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("cannot crop to larger canvas")
  }
  try
    @minimax_h3_video_decode.VideoOutput::new(
      frames=0,
      input_height=64,
      input_width=80,
      height=60,
      width=72,
      symbol="canonical",
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("empty frames accepted")
  }
}

///|
test "empty video request cannot submit or publish and releases idempotently" {
  let request = @minimax_h3_video_decode.VideoDecodeRequest::new()
  try request.submit() catch {
    _ => ()
  } noraise {
    _ => fail("empty submit")
  }
  try request.poll() catch {
    _ => ()
  } noraise {
    _ => fail("empty poll")
  }
  try request.wait() catch {
    _ => ()
  } noraise {
    _ => fail("empty wait")
  }
  try request.with_output(_ => ()) catch {
    _ => ()
  } noraise {
    _ => fail("empty output")
  }
  request.cancel()
  request.cancel()
  try request.submit() catch {
    _ => ()
  } noraise {
    _ => fail("cancelled submit")
  }
  request.close()
  request.close()
  request.cancel()
  try request.with_output(_ => ()) catch {
    _ => ()
  } noraise {
    _ => fail("closed output")
  }
}

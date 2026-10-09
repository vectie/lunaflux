///|
test "video frontend retains all official tokens and F32 storage" {
  let frontend = @minimax_h3_video_decode.Frontend::new(
    batch=1,
    temporal=37,
    height=48,
    width=84,
    prefix="video_front",
    max_workspace_bytes=2000000000L,
  )
  assert_eq(frontend.rows_per_batch, 149189)
  assert_eq(frontend.input_bytes, 14321664L)
  assert_eq(frontend.output_bytes, 1222156288L)
  assert_eq(frontend.weights.byte_count, 239968L)
  assert_eq(frontend.symbols.length(), 2)
  let source = @utf8.decode(frontend.source)
  assert_true(source.contains("video_front_post_quant"))
  assert_true(source.contains("video_front_embedding"))
  assert_true(source.contains("row < 149188ULL"))
  assert_true(source.contains("float *output"))
  assert_false(source.contains("bfloat16"))
}

///|
test "frontend budget includes both intermediate and persistent output" {
  let frontend = @minimax_h3_video_decode.Frontend::new(
    batch=2,
    temporal=1,
    height=2,
    width=3,
    prefix="tile",
    max_workspace_bytes=1000000L,
  )
  let total = frontend.input_bytes + frontend.output_bytes
  let exact = @minimax_h3_video_decode.Frontend::new(
    batch=2,
    temporal=1,
    height=2,
    width=3,
    prefix="tile",
    max_workspace_bytes=total,
  )
  assert_eq(exact.output_bytes, frontend.output_bytes)
  try
    @minimax_h3_video_decode.Frontend::new(
      batch=2,
      temporal=1,
      height=2,
      width=3,
      prefix="tile",
      max_workspace_bytes=total - 1L,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("expected workspace rejection")
  }
  try
    @minimax_h3_video_decode.Frontend::new(
      batch=1,
      temporal=2147483647,
      height=2147483647,
      width=2,
      prefix="tile",
      max_workspace_bytes=total,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("expected overflow rejection")
  }
}

///|
test "frontend unprepared owner rejects binding access and closes idempotently" {
  let frontend = @minimax_h3_video_decode.Frontend::new(
    batch=1,
    temporal=1,
    height=1,
    width=1,
    prefix="tile",
    max_workspace_bytes=100000L,
  )
  let program = @minimax_h3_video_decode.FrontendProgram::new(frontend)
  try program.with_output(_ => 1) catch {
    _ => ()
  } noraise {
    _ => fail("unprepared output")
  }
  program.close()
  program.close()
}

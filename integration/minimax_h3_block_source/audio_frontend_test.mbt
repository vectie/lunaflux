///|
test "audio decode frontend exports projection and normalized convolution" {
  let source = @minimax_h3_block_source.build_audio_frontend(
    batch=2,
    time=207,
    function_prefix="audio_front",
  )
  assert_eq(source.symbols.length(), 3)
  assert_eq(source.symbols[0], "audio_front_project")
  assert_eq(source.symbols[2], "audio_front_conv")
  assert_eq(source.batch, 2)
  assert_eq(source.time, 207)
  assert_true(source.source.length() > 0)
}

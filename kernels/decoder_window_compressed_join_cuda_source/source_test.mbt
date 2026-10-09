///|
test "window-compressed join source fixes ring padding and literal order" {
  let rendered = deterministic_window_compressed_join_i32_source(
    window_size=128,
    compression_ratio=128,
    maximum_position_count=1048576,
    maximum_rows=32,
    function_symbol="lunaflux_decoder_window_compressed_join_i32_v1",
  )
  assert_eq(rendered.maximum_compressed_width(), 8192)
  assert_eq(rendered.maximum_joined_width(), 8320)
  let source = @utf8.decode(rendered.source_bytes())
  assert_true(source.contains("mode == 0 ? sequence_length / 128"))
  assert_true(
    source.contains("const int32_t first = (position % 128 + 1) % 128"),
  )
  assert_true(source.contains("compressed_base_value + slot"))
  assert_true(source.contains("window_width + (unsigned long long)slot"))
  assert_false(source.contains("sort"))
  assert_false(source.contains("dedup"))
}

///|
test "window-compressed join source rejects hostile bounds" {
  try
    deterministic_window_compressed_join_i32_source(
      window_size=128,
      compression_ratio=1,
      maximum_position_count=1048576,
      maximum_rows=32,
      function_symbol="lunaflux_decoder_window_compressed_join_i32_v1",
    )
  catch {
    InvalidCompressionRatio => ()
    _ => fail("unexpected ratio rejection")
  } noraise {
    _ => fail("unit compression ratio admitted")
  }
  try
    deterministic_window_compressed_join_i32_source(
      window_size=128,
      compression_ratio=128,
      maximum_position_count=1048576,
      maximum_rows=32,
      function_symbol="bad-symbol",
    )
  catch {
    InvalidFunctionSymbol => ()
    _ => fail("unexpected symbol rejection")
  } noraise {
    _ => fail("invalid symbol admitted")
  }
}

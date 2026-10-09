///|
test "video block stem respects mixed precision and partial 3D rotary" {
  let stem = @minimax_h3_video_decode.AttentionStem::new(
    temporal=2,
    height=3,
    width=4,
    layer=0,
    prefix="video_stem",
    max_scratch_bytes=1000000L,
  )
  assert_eq(stem.rows, 29)
  assert_eq(stem.hidden_bytes, 118784L)
  assert_eq(stem.qkv_bytes, 356352L)
  assert_eq(stem.table_bytes, 2784L)
  assert_eq(stem.weights.byte_count, 50364416L)
  assert_eq(stem.weights.copies.length(), 7)
  assert_true(stem.qkv_offset >= stem.hidden_bytes)
  assert_true(stem.table_offset >= stem.qkv_offset + stem.qkv_bytes)
  let source = @utf8.decode(stem.source)
  assert_true(source.contains("float values[64]"))
  assert_true(source.contains("if(c<48)"))
  assert_true(source.contains("__float2bfloat16_rn(weight"))
  assert_true(source.contains("plane == 2"))
  assert_true(source.contains("position = 0.0f"))
  assert_true(source.contains("row < 24ULL"))
  assert_true(source.contains("const int axis = pair / 8"))
}

///|
test "video attention stem rejects unavailable layer and short scratch" {
  for layer in [-1, 36] {
    try
      @minimax_h3_video_decode.AttentionStem::new(
        temporal=1,
        height=1,
        width=1,
        layer~,
        prefix="video_stem",
        max_scratch_bytes=1000000L,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("invalid layer")
    }
  }
  try
    @minimax_h3_video_decode.AttentionStem::new(
      temporal=1,
      height=1,
      width=1,
      layer=0,
      prefix="video_stem",
      max_scratch_bytes=1L,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("invalid scratch")
  }
}

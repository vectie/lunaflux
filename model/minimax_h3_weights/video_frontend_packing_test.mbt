///|
test "video frontend packing matches official five tensor schema" {
  let layout = @minimax_h3_weights.video_frontend_weight_packing(
    max_bytes=239968L,
  )
  assert_eq(layout.byte_count, 239968L)
  assert_eq(layout.copies.length(), 5)
  assert_eq(layout.copies[0].tensor_name, "post_quant_conv.weight")
  assert_eq(layout.copies[0].byte_count, 2304L)
  assert_eq(layout.copies[1].destination_offset, 2304L)
  assert_eq(layout.copies[2].destination_offset, 2400L)
  assert_eq(layout.copies[3].destination_offset, 199008L)
  assert_eq(layout.copies[4].destination_offset, 207200L)
  assert_eq(layout.copies[4].byte_count, 32768L)
  try
    @minimax_h3_weights.video_frontend_weight_packing(max_bytes=239967L)
  catch {
    _ => ()
  } noraise {
    _ => fail("expected weight budget rejection")
  }
}

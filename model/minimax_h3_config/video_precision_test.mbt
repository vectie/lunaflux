///|
test "original video decoder autocast is FP16 independently of text BF16" {
  assert_eq(@minimax_h3_config.video_decoder_autocast(), Float16)
}

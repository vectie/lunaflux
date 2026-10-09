///|
test "official decoder packing matches all upsamplers and bias-free tail" {
  let frontend = audio_decoder_weight_packing(Frontend, max_bytes=58998784L)
  assert_eq(frontend.byte_count, 58998784L)
  assert_eq(frontend.copies.length(), 5)
  assert_eq(frontend.copies[2].destination_offset, 270336L)
  assert_eq(frontend.copies[3].destination_offset, 274432L)
  assert_eq(frontend.copies[4].destination_offset, 58994688L)
  let tail = audio_decoder_weight_packing(WaveformTail, max_bytes=388L)
  assert_eq(tail.byte_count, 388L)
  assert_eq(tail.copies.length(), 6)
  assert_eq(tail.copies[4].destination_offset, 160L)
  assert_eq(tail.copies[5].destination_offset, 164L)
  assert_eq(tail.copies[5].tensor_name, "decoder.conv_post.weight_v")
  for stage in 0..<7 {
    let plan = audio_decoder_weight_packing(
      Upsample(stage),
      max_bytes=100000000L,
    )
    let ci = 1024L >> stage
    let co = ci / 2L
    let kernel = if stage < 2 { 9L } else { 4L }
    assert_eq(plan.copies.length(), 3)
    assert_eq(plan.copies[0].byte_count, ci * 4L)
    assert_eq(plan.copies[1].destination_offset, ci * 4L)
    assert_eq(plan.copies[1].byte_count, ci * co * kernel * 4L)
    assert_eq(plan.copies[2].byte_count, co * 4L)
    ignore(
      audio_decoder_weight_packing(Upsample(stage), max_bytes=plan.byte_count),
    )
  }
  let invalid : Array[AudioDecoderWeightGroup] = [
    WaveformTail,
    Upsample(-1),
    Upsample(7),
  ]
  for group in invalid {
    try audio_decoder_weight_packing(group, max_bytes=387L) catch {
      _ => ()
    } noraise {
      _ => fail("invalid packing accepted")
    }
  }
}

///|
test "audio stage joins upsample and AMP geometry under one budget" {
  let up = @minimax_h3_block_source.build_audio_upsample(
    batch=2,
    input_channels=16,
    output_channels=8,
    input_time=31,
    kernel=4,
    stride=2,
    padding=1,
    function_prefix="up",
  )
  let amp = @minimax_h3_block_source.build_audio_amp(
    batch=2,
    channels=8,
    time=62,
    function_prefix="amp",
  )
  let layout = audio_stage_layout(up, amp, 1000000L)
  let total = layout.local_bytes + layout.amp_bytes
  ignore(audio_stage_layout(up, amp, total))
  assert_true(layout.normalized_offset >= layout.upsample.output_bytes)
  try audio_stage_layout(up, amp, total - 1L) catch {
    _ => ()
  } noraise {
    _ => fail("combined stage budget not enforced")
  }
  let wrong = @minimax_h3_block_source.build_audio_amp(
    batch=2,
    channels=8,
    time=31,
    function_prefix="wrong",
  )
  try audio_stage_layout(up, wrong, 1000000L) catch {
    _ => ()
  } noraise {
    _ => fail("mismatched AMP time accepted")
  }
}

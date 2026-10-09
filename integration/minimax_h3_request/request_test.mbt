///|
test "unprepared joint request cannot publish either modality or advance" {
  let request = @minimax_h3_request.Request::new()
  assert_true(!request.outputs_ready())
  try request.advance() catch {
    _ => ()
  } noraise {
    _ => fail("unprepared request advanced")
  }
  try request.with_audio(_ => 1) catch {
    _ => ()
  } noraise {
    _ => fail("uninitialized audio published")
  }
  try request.with_video(_ => 1) catch {
    _ => ()
  } noraise {
    _ => fail("uninitialized video published")
  }
  try request.waveform_samples() catch {
    _ => ()
  } noraise {
    _ => fail("uninitialized waveform size published")
  }
  request.cancel()
  request.close()
  request.close()
  assert_true(!request.outputs_ready())
}

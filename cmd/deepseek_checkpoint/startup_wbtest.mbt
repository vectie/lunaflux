///|
test "DeepSeek diagnostic request frame retains absolute positions and model vocabulary" {
  let frame = request_frame(3, 256, 129280)
  assert_eq(frame.input_count(), 9)
  assert_eq(frame.geometry(), (3, 256, 129280))
  let (protocol, _) = prompt_limits(3, 256, 129280)
  assert_eq(protocol.max_plan_tokens(), 3)
  assert_eq(protocol.max_token_id(), 129279)
}

///|
test "DeepSeek diagnostic frame rejects context smaller than live-row capacity" {
  try request_frame(4, 3, 129280) catch {
    @frame.SerialDecoderError::InvalidCapacity => ()
    error => raise error
  } noraise {
    _ => fail("unbounded/inconsistent request geometry")
  }
}

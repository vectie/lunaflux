///|
test "step limits are narrow and bounded" {
  let limits = @device_step.DeviceStepLimits::new(
    max_rows=8,
    max_tokens=64,
    max_pages=32,
    max_sequence_tokens=2048,
    max_token_id=32000,
  )
  assert_eq(limits.max_rows(), 8)
  assert_eq(limits.max_tokens(), 64)
  assert_eq(limits.max_pages(), 32)
  assert_eq(limits.max_sequence_tokens(), 2048)
  assert_eq(limits.max_token_id(), 32000)
  try
    @device_step.DeviceStepLimits::new(
      max_rows=0,
      max_tokens=1,
      max_pages=1,
      max_sequence_tokens=1,
      max_token_id=0,
    )
  catch {
    InvalidLimit(Rows) => ()
    _ => fail("unexpected limit failure")
  } noraise {
    _ => fail("expected rows rejection")
  }
  try
    @device_step.DeviceStepLimits::new(
      max_rows=1,
      max_tokens=1,
      max_pages=1,
      max_sequence_tokens=0,
      max_token_id=0,
    )
  catch {
    InvalidLimit(SequenceTokens) => ()
    _ => fail("unexpected sequence limit failure")
  } noraise {
    _ => fail("expected sequence limit rejection")
  }
  try
    @device_step.DeviceStepLimits::new(
      max_rows=1,
      max_tokens=1,
      max_pages=1,
      max_sequence_tokens=1,
      max_token_id=-1,
    )
  catch {
    InvalidLimit(TokenId) => ()
    _ => fail("unexpected token limit failure")
  } noraise {
    _ => fail("expected token limit rejection")
  }
}

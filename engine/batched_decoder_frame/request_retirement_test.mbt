///|
test "never-submitted cancellation retires without resetting a live neighbor" {
  let host = host()
  host.retire_request(90UL, 0UL)
  let (_, input, writer) = plan([(10UL, 0UL, 0, [1], false, true, false)])
  host.stage(input, writer)
  try host.retire_request(90UL, 0UL) catch {
    @batched_decoder_frame.BatchFrameError::Busy => ()
    e => raise e
  } noraise {
    _ => fail("release while execution owns slots")
  }
  ignore(host.complete([2], execution_failed=false))
  host.retire_request(90UL, 0UL)
  assert_eq(host.processed_tokens(10UL, 0UL), 1)
  try host.retire_request(10UL, 1UL) catch {
    @batched_decoder_frame.BatchFrameError::GenerationMismatch => ()
    e => raise e
  } noraise {
    _ => fail("wrong generation released history")
  }
  host.retire_request(10UL, 0UL)
  host.retire_request(10UL, 0UL)
  assert_false(host.contains_request(10UL, 0UL))
  let (_, input, writer) = plan([(10UL, 1UL, 0, [3], false, true, false)])
  host.stage(input, writer)
  assert_eq(port(host, RecurrentResets), [1])
  ignore(host.complete([4], execution_failed=false))
  try host.retire_request(10UL, 0UL) catch {
    @batched_decoder_frame.BatchFrameError::GenerationMismatch => ()
    e => raise e
  } noraise {
    _ => fail("late cancellation released reused slot")
  }
  assert_eq(host.processed_tokens(10UL, 1UL), 1)
  host.close()
}

///|
test "direct completion writer reserves an abort epoch before mutation" {
  let plan_owner = PlanFrameBuffer::new(
    completion_test_owner().worker_limits,
    completion_test_owner().inference_limits,
  )
  let bytes = valid_two_decode_frame()
  let plan = plan_owner.load(bytes, bytes.length())
  let completion = completion_test_owner()
  completion.epoch = 0xfffffffffffffffeUL
  try completion.writer_for(plan) catch {
    FrameEpochOverflow => ()
    _ => fail("expected terminal writer epoch rejection")
  } noraise {
    _ => fail("terminal writer epoch opened")
  }
  assert_false(completion.writer_open)
  assert_eq(completion.length, 0)
  assert_eq(completion.entry_count, 0)
}

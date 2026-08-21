///|
test "tokenizer input begin rejection releases lane without restoring receipt" {
  let now = Ref(11UL)
  let pool = fixture_preparation_pool(now, 1, 1, 4096UL)
  let received = pool_received_at(fixture_request(fixture_identity("0")), 10UL)
  let lane_index = pool.free_ring[pool.free_head]
  let blocker = pool.lanes[lane_index].tokenizer.begin_luna_input(
    byte_count=0,
    special_policy=Reject,
    output_limit=1,
    truncation=RejectOverflow,
  )
  try ignore(pool.try_submit(received)) catch {
    Invalid(Tokenization) => ()
    _ => fail("unexpected tokenizer-input begin rejection")
  } noraise {
    _ => fail("busy tokenizer input writer was replaced")
  }
  assert_true(received.consumed)
  assert_eq(pool.free_lane_count(), 1)
  assert_eq(pool.lanes[lane_index].phase, LUNA_PREPARATION_IDLE)
  blocker.abort()
  let replacement = pool_received_at(
    fixture_request(fixture_identity("0")),
    10UL,
  )
  let work = pool.try_submit(replacement).work()
  assert_true(replacement.consumed)
  work.cancel()
  ignore(pool.progress())
  assert_eq(pool.free_lane_count(), 1)
}

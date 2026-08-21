///|
test "Luna direct receipt exposes only authenticated remaining wait" {
  let (source, _) = framed_pool_source(fixture_request(fixture_identity("0")))
  let now = Ref(10UL)
  let pool = direct_hostile_pool(() => now.val, 1, 1, 100000UL)
  let work = pool
    .try_begin_luna_framed(source, source_offset=0, length=1)
    .work()
  let maximum = fixture_inference().max_deadline_millis()
  assert_eq(work.luna_framed_receipt_remaining_millis(), maximum.to_int())
  now.val = 10UL + maximum - 1UL
  assert_eq(work.luna_framed_receipt_remaining_millis(), 1)
  now.val = 10UL + maximum
  assert_eq(work.luna_framed_receipt_remaining_millis(), 0)
  work.cancel()
  ignore(pool.progress())
  try work.luna_framed_receipt_remaining_millis() catch {
    Invalid(LunaPreparedLifecycle) => ()
    _ => fail("unexpected stale Luna receipt wait error")
  } noraise {
    _ => fail("stale Luna receipt exposed a deadline interval")
  }
}

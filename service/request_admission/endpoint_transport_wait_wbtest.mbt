///|
test "Luna endpoint receipt wake preserves exact 2 1 0 and clock failure" {
  let (source, _) = framed_pool_source(fixture_request(fixture_identity("0")))
  let now = Ref(10UL)
  let fail_clock = Ref(false)
  let pool = direct_hostile_pool(
    () => {
      if fail_clock.val {
        raise Invalid(ReceiptTime)
      }
      now.val
    },
    1,
    1,
    100000UL,
  )
  let work = pool
    .try_begin_luna_framed(source, source_offset=0, length=1)
    .work()
  let maximum = fixture_inference().max_deadline_millis()
  now.val = 10UL + maximum - 2UL
  assert_eq(work.luna_framed_receipt_remaining_millis(), 2)
  now.val += 1UL
  assert_eq(work.luna_framed_receipt_remaining_millis(), 1)
  now.val += 1UL
  assert_eq(work.luna_framed_receipt_remaining_millis(), 0)
  now.val = 9UL
  try work.luna_framed_receipt_remaining_millis() catch {
    Invalid(ReceiptTime) => ()
    _ => fail("unexpected backward Luna receipt clock error")
  } noraise {
    _ => fail("backward Luna receipt clock enlarged its wait interval")
  }
  now.val -= 1UL
  fail_clock.val = true
  try work.luna_framed_receipt_remaining_millis() catch {
    Invalid(ReceiptTime) => ()
    _ => fail("unexpected Luna endpoint receipt clock error")
  } noraise {
    _ => fail("failed Luna receipt clock exposed a wait interval")
  }
}
